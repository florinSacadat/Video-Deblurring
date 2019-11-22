from pytube import YouTube
# yt = YouTube("https://www.youtube.com/watch?v=LXb3EKWsInQ")
# yt = yt('mp4', '2160p')
# yt.download('/home/student/Documents/YoutubePymp4/Saved')
YouTube('https://www.youtube.com/watch?v=krSxqVgNZyo').streams.first().download('/home/student/Documents/YoutubePymp4/Saved')