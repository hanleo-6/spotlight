import os
import pandas as pd
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import pickle

SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
FOLDER_ID = '13xgQN3ou1lOW1_P_yj2f2J5X13vjtUfM'

def authenticate():
    creds = None

    if os.path.exists('thumbnail_extraction/token.pickle'):
        with open('thumbnail_extraction/token.pickle', 'rb') as token:
            creds = pickle.load(token)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES
            )
            creds = flow.run_local_server(port=0)

        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)

    return build('drive', 'v3', credentials=creds)

def fetch_videos(service):
    results = []
    page_token = None

    while True:
        response = service.files().list(
            q=f"'{FOLDER_ID}' in parents and mimeType contains 'video/'",
            fields="nextPageToken, files(id, name, thumbnailLink)",
            pageToken=page_token
        ).execute()

        for file in response.get('files', []):
            # thumb = file.get('thumbnailLink', '')
            
            # # Higher-res thumbnail
            # if thumb:
            #     thumb = thumb.replace('=s220', '=s600')

            thumb_url = f"https://drive.googleusercontent.com/uc?id={file['id']}"


            results.append({
                'Video Name': file['name'],
                'Thumbnail': f'=IMAGE("{thumb_url}")' if thumb_url else ''
            })

        page_token = response.get('nextPageToken', None)
        if not page_token:
            break

    return results

def main():
    service = authenticate()
    videos = fetch_videos(service)

    df = pd.DataFrame(videos)
    df.to_csv('video_thumbnails.csv', index=False)

    print(f"Saved {len(df)} videos to video_thumbnails.csv")

if __name__ == '__main__':
    main()
