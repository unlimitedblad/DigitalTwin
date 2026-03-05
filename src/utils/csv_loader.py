"""
微信聊天记录 CSV 加载器
"""

import os
from pathlib import Path
from datetime import datetime
from typing import List, Set, Tuple, Optional

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc="Processing", total=None):
        print(f"{desc}...")
        return iterable

from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_core.documents import Document


class WeChatCSVLoader:
    """自定义微信聊天记录CSV加载器"""

    def __init__(self, csv_folder_path, encoding="utf-8"):
        self.csv_folder_path = Path(csv_folder_path)
        self.encoding = encoding

    def load(self, incremental=False, tracking_data=None, csv_pattern: str = None) -> Tuple[List[Document], Set[str]]:
        """加载所有CSV文件并返回文档列表

        Args:
            incremental: 是否为增量更新模式
            tracking_data: 导入跟踪数据（增量模式下使用）
            csv_pattern: glob 匹配模式，覆盖 CSV_FILE_PATTERN 环境变量

        Returns:
            (documents, new_hashes)
        """
        from .tracking import generate_record_hash

        documents = []
        new_hashes = set()
        skipped_count = 0

        if tracking_data is None:
            tracking_data = {"imported_hashes": set(), "file_timestamps": {}}
        else:
            tracking_data["imported_hashes"] = set(tracking_data.get("imported_hashes", []))

        pattern = csv_pattern or os.getenv("CSV_FILE_PATTERN", "*.csv")
        csv_files = list(self.csv_folder_path.glob(pattern))
        print(f"找到 {len(csv_files)} 个CSV文件（匹配: {pattern}）")

        for csv_file in tqdm(csv_files, desc="处理CSV文件"):
            print(f"正在处理: {csv_file.name}")

            loader = CSVLoader(
                file_path=str(csv_file),
                encoding=self.encoding,
                csv_args={'delimiter': ','},
                metadata_columns=['CreateTime', 'talker', 'msg', 'type_name', 'room_name', 'is_sender'],
            )

            try:
                file_docs = loader.load()
                processed_count = 0
                valid_count = 0

                for doc in file_docs:
                    try:
                        msg_content = doc.metadata.get('msg', '').strip()
                        if not msg_content:
                            continue

                        raw_time = doc.metadata.get('CreateTime', '')
                        talker = doc.metadata.get('talker', '')
                        type_name = doc.metadata.get('type_name', '')
                        room_name = doc.metadata.get('room_name', '')
                        is_sender = doc.metadata.get('is_sender', '0')
                        row = doc.metadata.get('row', 0)

                        if incremental:
                            record_hash = generate_record_hash(
                                str(csv_file.name), row, raw_time, msg_content
                            )
                            if record_hash in tracking_data["imported_hashes"]:
                                skipped_count += 1
                                continue
                            new_hashes.add(record_hash)

                        if (len(msg_content) <= 2 or
                                msg_content.startswith('[') or
                                msg_content.startswith('表情') or
                                '动画表情' in type_name or
                                msg_content == "I've accepted your friend request. Now let's chat!" or
                                '<msg>' in msg_content):
                            continue

                        # 精简格式：去除冗余字段，保留对嵌入有价值的信息
                        location = f"@{room_name}" if room_name else ""
                        formatted_content = f"{talker}{location}: {msg_content}"

                        chat_timestamp = 0
                        if raw_time:
                            try:
                                chat_timestamp = int(datetime.fromisoformat(raw_time).timestamp())
                            except Exception:
                                try:
                                    chat_timestamp = int(datetime.strptime(raw_time, "%Y-%m-%d %H:%M:%S").timestamp())
                                except Exception:
                                    chat_timestamp = 0

                        new_doc = Document(
                            page_content=formatted_content,
                            metadata={
                                "source": str(csv_file.name),
                                "chat_time": chat_timestamp,
                                "chat_time_str": raw_time,
                                "sender": talker,
                                "msg_type": type_name,
                                "room": room_name,
                                "is_sender": is_sender,
                                "msg_content": msg_content[:200],
                            }
                        )
                        documents.append(new_doc)
                        valid_count += 1

                    except Exception:
                        continue

                    processed_count += 1

                print(f"  - 处理了 {processed_count} 条记录，有效记录 {valid_count} 条")

            except Exception as e:
                print(f"处理文件 {csv_file} 时出错: {e}")
                continue

        if incremental and skipped_count > 0:
            print(f"\n增量更新: 跳过 {skipped_count} 条已导入的记录")

        return documents, new_hashes
