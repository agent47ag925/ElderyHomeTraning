import os
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import pandas as pd

# API 키 설정
# Google Cloud Console에서 발급받은 본인의 API 키를 여기에 입력하세요.
API_KEY = "AIzaSyDNu0khm-DSEBVKLVqILs85aNrQfPQDdRw"

# YouTube API 클라이언트 생성
youtube = build('youtube', 'v3', developerKey=API_KEY)

def search_and_get_description(keyword, max_results=5):
    try:
        # 1) 검색
        search_response = youtube.search().list(
            q=keyword,
            part='id,snippet',
            maxResults=max_results,
            type='video'
        ).execute()

        video_ids = [item['id']['videoId'] for item in search_response.get('items', [])]
        if not video_ids:
            print(f"'{keyword}'에 대한 검색 결과가 없습니다.")
            return

        # 2) 상세(snippet) 조회
        videos_response = youtube.videos().list(
            id=','.join(video_ids),
            part='snippet, contentDetails'
        ).execute()

        snip = {}
        count = 0
        
        #print(f"✅ '{keyword}'에 대한 동영상 검색 결과:\n")
        for item in videos_response.get('items', []):
            ids = item['id']
            title = item['snippet']['title']
            description = item['snippet'].get('discription', '정보 없음')
            published_date = item['snippet']['publishedAt']
            duration = item['contentDetails'].get('duration', '정보 없음')
            video_url = f"https://www.youtube.com/watch?v={ids}"
            print("--------------------------------------------------")
            print(f"제목: {title}")
            print(f"설명:\n{description}")
            print(f'업로드일 {published_date}')
            print(f'지속성 {duration}')
            print("--------------------------------------------------\n")
            snip[str(count)] = [title, description, published_date, duration, video_url]
            count += 1

        return snip

    except HttpError as e:
        print(f"❌ API 호출 중 오류: {e}")
        print("API 키와 YouTube Data API v3 활성화 여부를 확인하세요.")
        return e

    except Exception as e:
        print(f"❌ 예기치 않은 오류: {e}")
        return e


if __name__ == "__main__":
    word = input('입력')
    search_and_get_description(word)
