"""
YouTube 검색 시 query 작성에 대한 주요 규칙과 팁:

1. 기본 검색:
   - 단순히 키워드나 문구를 입력합니다.
   - 예: `Python tutorial`

2. 정확한 구문 검색:
   - 큰따옴표로 묶어 정확한 구문을 검색합니다.
   - 예: `"Python for beginners"`

3. 불리언 연산자:
   - AND: 기본적으로 모든 단어는 AND로 연결됩니다.
   - OR: 단어 사이에 OR을 사용합니다.
   - 예: `Python OR Java tutorial`

4. 제외 검색:
   - 빼고 싶은 단어 앞에 마이너스(-) 기호를 사용합니다.
   - 예: `Python tutorial -advanced`

5. 와일드카드:
   - 별표(*)를 사용해 부분 일치를 검색할 수 있습니다.
   - 예: `Python * tutorial`

6. 채널 검색:
   - 채널명 앞에 @를 붙입니다.
   - 예: `@GoogleDevelopers Python`

7. 필터 사용:
   - 특정 필터를 쿼리에 직접 포함할 수 있습니다.
   - 예: `Python tutorial after:2023-01-01` (2023년 이후 영상)

8. 대소문자 구분:
   - YouTube 검색은 대소문자를 구분하지 않습니다.

9. 특수 문자:
   - 대부분의 특수 문자는 무시됩니다.

10. 길이 제한:
    - 쿼리 문자열에는 길이 제한이 있으므로 너무 길지 않게 작성합니다.

11. 언어와 지역:
    - 검색 결과는 사용자의 위치와 언어 설정에 영향을 받을 수 있습니다.

12. 동의어:
    - YouTube는 때때로 동의어를 자동으로 포함시킵니다.

이러한 규칙을 조합하여 더 정확하고 효과적인 검색 쿼리를 만들 수 있습니다. 예를 들어:

```python
query = '"Python for beginners" -advanced after:2023-01-01'
```

이 쿼리는 "Python for beginners"라는 정확한 구문을 포함하고, "advanced"를 제외하며, 2023년 1월 1일 이후에 업로드된 비디오를 검색합니다.

"""
from typing import List, Tuple, Dict, Any, Union, Optional
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import os
import logging

# SSL 인증을 비활성화한 HTTP 객체 생성
import httplib2

class YouTubeAPIKeyError(Exception):
    pass

class YouTube:

    def __init__(self, developerKey: str=None) -> None:
        
        if not developerKey:
            try:
                developerKey = os.environ["YOUTUBE_DEVELOPER_KEY"]
            except KeyError:
                raise YouTubeAPIKeyError("YouTube Developer Key가 설정되지 않았습니다. 환경 변수 'YOUTUBE_DEVELOPER_KEY'를 설정하거나 직접 키를 제공해주세요.")

        http = httplib2.Http(disable_ssl_certificate_validation=True)
        self.youtube = build("youtube", "v3", developerKey=developerKey, http=http)

    def search(self,
            query: str,
            max_results: int = 5,
            caption: Optional[bool] = None,
            caption_language: Optional[str] = None
            ) -> List[Dict]:
        
        search_params = {
            'q': query,
            'type': 'video',
            'part': 'id,snippet',
            'maxResults': max_results * 2  # 더 많은 결과를 가져와 필터링 후 충분한 결과를 얻을 수 있도록 함
        }
        
        if caption is True:
            search_params['videoCaption'] = 'closedCaption'
        elif caption is False:
            search_params['videoCaption'] = 'none'
        
        search_response = self.youtube.search().list(**search_params).execute()
        
        results = []
        for item in search_response.get('items', []):
            video_id = item['id']['videoId']
            
            # 자막 정보 가져오기
            captions_response = self.youtube.captions().list(
                part='snippet',
                videoId=video_id
            ).execute()
            
            # 지정된 언어의 자막 확인
            has_specified_caption = any(
                caption['snippet']['language'] == caption_language 
                for caption in captions_response.get('items', [])
            ) if caption_language else True
            
            # caption이 True이고 지정된 언어의 자막이 없으면 건너뛰기
            if caption is True and not has_specified_caption:
                continue
            
            video_data = {
                'id': video_id,
                'title': item['snippet']['title'],
                'description': item['snippet']['description'],
                'thumbnail': item['snippet']['thumbnails']['default']['url'],
                'publishedAt': item['snippet']['publishedAt'],
                'channel': item['snippet']['channelTitle'],
                'has_specified_caption': has_specified_caption,
            }
            results.append(video_data)
            
            if len(results) == max_results:
                break
        
        return results
    
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class Content:
    id: str
    title: str
    description: str
    thumbnail: str
    published_at: datetime
    has_specified_caption: bool
    channel: str
    captions: List[Dict] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            id=data['id'],
            title=data['title'],
            description=data['description'],
            thumbnail=data['thumbnail'],
            published_at=datetime.fromisoformat(data['publishedAt'].rstrip('Z')),
            channel=data['channel'],
            has_specified_caption=data['has_specified_caption']
        )

    def to_dict(self):
        return {
            'id': self.id,
            'title': self.title,
            'description': self.description,
            'thumbnail': self.thumbnail,
            'publishedAt': self.published_at.isoformat() + 'Z',
            'has_specified_caption': self.has_specified_caption,
            'channel': self.channel,
            'captions': self.captions
        }

    def add_captions(self, captions: List[Dict]):
        self.captions.extend(captions)

class ContentManager:
    def __init__(self):
        self._contents: Dict[str, Content] = {}

    def add(self, content: Content):
        self._contents[content.id] = content

    def get(self, id: str) -> Optional[Content]:
        return self._contents.get(id)

    def remove(self, id: str):
        self._contents.pop(id, None)

    def all(self) -> List[Content]:
        return list(self._contents.values())

    def add_captions(self, id: str, captions: List[Dict]):
        content = self.get(id)
        if content:
            content.add_captions(captions)
        else:
            raise KeyError(f"Content with id '{id}' not found")

    def __getitem__(self, id: str) -> Content:
        return self.get(id)

    def __len__(self):
        return len(self._contents)

    def __iter__(self):
        return iter(self._contents.values())