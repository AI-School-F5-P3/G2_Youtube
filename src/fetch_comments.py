from googleapiclient.discovery import build


def fetch_comments(video_url, api_key):
    '''
    Extract comments from a YouTube video.

    Parameters:
        video_url (str): The URL of the YouTube video.
        api_key (str): The API key for the YouTube Data API.

    Returns:
        list: A list of comments.
    '''
    # obtain the video ID from the URL
    video_id = video_url.split('v=')[1]
    if not video_id:
        raise ValueError("Invalid YouTube URL")

    # build the service
    youtube = build("youtube", "v3", developerKey=api_key)

    # make the request
    comments = []
    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=100,
    )

    while request:
        response = request.execute()
        for item in response["items"]:
            comment = item["snippet"]["topLevelComment"]["snippet"]
            ["textDisplay"]
            comments.append(comment)
        request = youtube.commentThreads().list_next(request, response)
    return comments
