from pytube import YouTube

# where to save
SAVE_PATH = "D:/Users/M/Music/"  # to_do

# link of the video to be downloaded
link = ["https://www.youtube.com/watch?v=8Vhd1rt8ZF0"
        ]


def Downloader():
    url = YouTube(str(link[0]))
    video = url.streams.first()
    video.download()


for i in link:
    try:

        # object creation using YouTube
        # which was imported in the beginning
        Downloader()
    except:

        # to handle exception
        print("Connection Error")

print('Task Completed!')
