import subprocess as sp
import os 
import argparse
import time


parser = argparse.ArgumentParser(formatter_class = argparse.RawDescriptionHelpFormatter)
parser.add_argument("-o", "--output_path")
parser.add_argument("-i", "--rootdirectory")
parser.add_argument("-f", "--sound_format", default = "mp3")
args = vars(parser.parse_args())

def check_and_extract(videofile, file, savedir, format):
    """ Extract .mp3 or .wav at 16Hz and .srt files from .mp4."""
    
    # Check if Swedish subtitles exist 
    out = sp.run(['ffprobe','-loglevel','error', "-select_streams", "s", "-show_entries", "stream=index:stream_tags=language", "-of", "csv=p=0", videofile], stdout=sp.PIPE, stderr=sp.PIPE, text=True, encoding="cp437").stdout.splitlines()

    # Check index of Swedish subtitles
    swe_idx = [i for i, x in enumerate(out) if "swe" in x]
    
    if len(swe_idx) >= 1:
        if os.path.isfile(f"{(os.path.join(savedir, file))[:-4]}.srt") == False:
            # Save subtitle in srt format
            out = sp.run(['ffmpeg','-i', videofile, '-map', f's:{swe_idx[0]}', '-f','srt', f'{savedir}/{file[:-4]}.srt'], stdout=sp.PIPE, stderr=sp.PIPE, universal_newlines=True)
        else:
            pass
        if format == "mp3":
        # Save sound in mp3 format
            out = sp.run(['ffmpeg', '-i' , videofile, '-acodec',  'libmp3lame', f'{savedir}/{file[:-4]}.mp3'])
        elif format == "wav":   
            out = sp.run(['ffmpeg', '-i' , videofile, '-acodec',  'pcm_s16le', '-ac', '1', '-ar', '16000', f'{savedir}/{file[:-4]}.wav'])
    else:
        print(f"Swedish subtitles are not available for {file}")
        

def main():

    rootdir = args["rootdirectory"]
    savedir = args["output_path"]
    format = args["sound_format"]
    
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            videofile_path = os.path.join(subdir, file)
            video_savedir = os.path.join(savedir, file)
            if os.path.isfile(f"{video_savedir[:-4]}.{format}") == False:
                check_and_extract(videofile_path, file, savedir, format)
            else:
              print(f"Skipping, {file[:-4]}.{format} already exists.")


if __name__ == "__main__":
    start = time.time()
    print(args)
    main()
    print(time.time() - start)



