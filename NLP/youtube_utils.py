from pytube import YouTube
import pandas as pd

def get_video_comments(video_url):
    try:
        yt = YouTube(video_url)
        comments = []
        for item in yt.initial_data['contents']['twoColumnBrowseResultsRenderer']['tabs'][0]['tabRenderer']['content']['sectionListRenderer']['contents']:
            if "commentThreadRenderer" in item:
                comment = item['commentThreadRenderer']['comment']['commentRenderer']['text']['runs'][0]['text']
                comments.append(comment)
        return comments
    except Exception as e:
        print(f"Error fetching comments: {e}")
        return []

