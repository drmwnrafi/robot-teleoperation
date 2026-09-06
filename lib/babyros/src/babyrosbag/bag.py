import json
import os
import time
import re
from collections import defaultdict

INDEX_PATTERN = re.compile(
    rb'"topic"\s*:\s*"((?:[^"\\]|\\.)*)"\s*,\s*'
    rb'"time"\s*:\s*\{\s*"secs"\s*:\s*(\d+)\s*,\s*"nsecs"\s*:\s*(\d+)\s*\}\s*,\s*'
    rb'"type"\s*:\s*"((?:[^"\\]|\\.)*)"'
)

class Time:
    """Mimics rospy.Time / babyros.Time"""
    def __init__(self, secs=0, nsecs=0):
        self.secs = secs
        self.nsecs = nsecs
        
    def to_sec(self):
        return self.secs + self.nsecs / 1e9

    @staticmethod
    def now():
        t = time.time()
        return Time(int(t), int((t - int(t)) * 1e9))
        
    def __lt__(self, other):
        return self.to_sec() < other.to_sec()

class Bag:
    """Mimics rosbag.Bag"""
    def __init__(self, filename, mode='r'):
        self.filename = filename
        self.mode = mode
        self.topics_info = defaultdict(lambda: {'count': 0, 'type': 'unknown'})
        self.start_time = None
        self.end_time = None
        
        if mode == 'w':
            self.file = open(filename, 'wb')  # Overwrite
        elif mode == 'a':
            self.file = open(filename, 'ab')  # Append
        elif mode == 'r':
            if not os.path.exists(filename):
                raise IOError(f"File {filename} not found")
            self.file = open(filename, 'rb')
            self._build_index()
        else:
            raise ValueError("Mode must be 'r', 'w', or 'a'")

    def _read_footer(self):
        """Attempts to read the index footer from the end of the file."""
        self.file.seek(0, 2)  # Seek to end
        file_size = self.file.tell()
        if file_size == 0:
            return False
            
        chunk_size = min(65536, file_size)
        self.file.seek(file_size - chunk_size)
        chunk = self.file.read(chunk_size)
        
        chunk = chunk.rstrip(b'\n\r')
        
        last_newline = chunk.rfind(b'\n')
        if last_newline != -1:
            last_line = chunk[last_newline+1:]
        else:
            last_line = chunk
            
        try:
            footer = json.loads(last_line)
            if footer.get("__bag_index__"):
                for topic, data in footer["topics"].items():
                    self.topics_info[topic] = data
                if footer["start_time"]:
                    self.start_time = Time(footer["start_time"]["secs"], footer["start_time"]["nsecs"])
                if footer["end_time"]:
                    self.end_time = Time(footer["end_time"]["secs"], footer["end_time"]["nsecs"])
                return True
        except Exception:
            pass
        return False

    def _build_index(self):
        """Reads the file to build topic info and time bounds."""
        if self._read_footer():
            return
            
        self.file.seek(0)
        topics_info = self.topics_info
        start_time = None
        end_time = None
        
        for line in self.file:
            match = INDEX_PATTERN.search(line)
            if not match:
                continue
                
            topic = match.group(1).decode('utf-8')
            t_secs = int(match.group(2))
            t_nsecs = int(match.group(3))
            msg_type = match.group(4).decode('utf-8')
            
            info = topics_info[topic]
            info['count'] += 1
            info['type'] = msg_type
            
            if start_time is None:
                start_time = (t_secs, t_nsecs)
                end_time = (t_secs, t_nsecs)
            else:
                if t_secs < start_time[0] or (t_secs == start_time[0] and t_nsecs < start_time[1]):
                    start_time = (t_secs, t_nsecs)
                if t_secs > end_time[0] or (t_secs == end_time[0] and t_nsecs > end_time[1]):
                    end_time = (t_secs, t_nsecs)
                    
        self.start_time = Time(start_time[0], start_time[1]) if start_time else None
        self.end_time = Time(end_time[0], end_time[1]) if end_time else None

    def write(self, topic, msg, t=None):
        if self.mode not in ('w', 'a'):
            raise IOError("Bag must be opened in 'w' or 'a' mode to write")
        
        if t is None:
            t = Time.now()
            
        if hasattr(msg, '__dict__'):
            msg_dict = msg.__dict__
        elif isinstance(msg, dict):
            msg_dict = msg
        else:
            msg_dict = {'data': msg}
            
        record = {
            'topic': topic,
            'time': {'secs': t.secs, 'nsecs': t.nsecs},
            'type': type(msg).__name__,
            'msg': msg_dict
        }
        
        line = json.dumps(record).encode('utf-8') + b'\n'
        self.file.write(line)
        
        self.topics_info[topic]['count'] += 1
        self.topics_info[topic]['type'] = type(msg).__name__
        
        if self.start_time is None or t < self.start_time:
            self.start_time = t
        if self.end_time is None or self.end_time < t:
            self.end_time = t

    def read_messages(self, topics=None, start_time=None, end_time=None):
        if self.mode != 'r':
            raise IOError("Bag must be opened in 'r' mode to read")
            
        self.file.seek(0)
        
        start_sec = start_time.to_sec() if start_time else None
        end_sec = end_time.to_sec() if end_time else None
        topics_set = set(topics) if topics else None
        
        loads = json.loads 
        
        for line in self.file:
            try:
                record = loads(line)
                if record.get("__bag_index__"):
                    continue
            except Exception:
                continue
                
            rec_topic = record['topic']
            if topics_set and rec_topic not in topics_set:
                continue
                
            rec_time_dict = record['time']
            rec_secs = rec_time_dict['secs']
            rec_nsecs = rec_time_dict['nsecs']
            
            rec_sec_float = rec_secs + rec_nsecs * 1e-9
            if start_sec is not None and rec_sec_float < start_sec:
                continue
            if end_sec is not None and rec_sec_float > end_sec:
                continue
                
            yield rec_topic, record['msg'], Time(rec_secs, rec_nsecs)

    def get_type_and_topic_info(self, topic_filters=None):
        class TopicInfo:
            def __init__(self, msg_type, message_count):
                self.msg_type = msg_type
                self.message_count = message_count
                
        class Info:
            def __init__(self):
                self.topics = {}
                
        info = Info()
        for topic, data in self.topics_info.items():
            if topic_filters and topic not in topic_filters:
                continue
            info.topics[topic] = TopicInfo(data['type'], data['count'])
            
        return info

    def close(self):
        if hasattr(self, 'file') and self.file:
            if self.mode in ('w', 'a'):
                footer = {
                    "__bag_index__": True,
                    "topics": dict(self.topics_info),
                    "start_time": {"secs": self.start_time.secs, "nsecs": self.start_time.nsecs} if self.start_time else None,
                    "end_time": {"secs": self.end_time.secs, "nsecs": self.end_time.nsecs} if self.end_time else None,
                }
                self.file.write(json.dumps(footer).encode('utf-8') + b'\n')
            self.file.close()
            
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()