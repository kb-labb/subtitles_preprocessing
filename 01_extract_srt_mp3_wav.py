import subprocess as sp
import os
from os import makedirs 
import pysrt
import argparse
import time
import csv

parser = argparse.ArgumentParser(formatter_class = argparse.RawDescriptionHelpFormatter)

parser.add_argument(
    "--o",
    type=str,
    default="results",
    help="Directory where wav and srt files end up.",
)
parser.add_argument(
    "--i",
    type=str,
    default="input",
    help="Directory containing mp4 files.",
)

parser.add_argument("-f", "--sound_format", default = "wav")

parser.add_argument(
    "--csv",
    type=str,
    default="subtitles_info.csv",
    help="Directory where csv file with info about subs will be saved.",
)

args = vars(parser.parse_args())

channels = ["cmore/sfkanalen"]
#channels = ["cmore/cmorefirst", "cmore/cmorehist", "cmore/cmoreseries", "cmore/cmorestars", "cmore/sfkanalen", \
            #"tv4/sjuan", "tv4/tv12", "tv4/tv4", "tv4/tv4fakta", "tv4/tv4film", \
            #"viasat/vfilmaction", "viasat/vfilmfamily", "viasat/vfilmhits", \
            #"viasat/vfilmpremiere", "viasat/vseries", "tv6/tv6", "tv8/tv8", "tv3/tv3"] 

def fuse_subtitles(fn_in: str, fn_out) -> None:
    subs = pysrt.open(fn_in)

    mysubs = []
    prev = None
    start = -1
    end = -1
    index = 0
    for s in subs:
        if s.text != prev:
            if prev is not None and end - start > 0:
                ns = pysrt.srtitem.SubRipItem(
                    start=start, end=end, text=prev, index=index
                )
                mysubs.append(ns)
            start = s.start
            end = s.end
            prev = s.text
            index += 1
        elif s.text == prev:
            end = s.end
    if prev is not None and end - start > 0:
        ns = pysrt.srtitem.SubRipItem(start=start, end=end, text=prev)
        mysubs.append(ns)

    new_subs = pysrt.SubRipFile(mysubs)

    new_subs.save(fn_out, encoding="utf-8")


def check_and_extract(videofile, file, savedir, format):
    """ Extract .mp3 or .wav at 16Hz and .srt files from .mp4."""

    # Check if Swedish subtitles exist 
    out = sp.run(['ffprobe','-loglevel','error', "-select_streams", "s", "-show_entries", "stream=index:stream_tags=language", "-of", "csv=p=0", videofile], stdout=sp.PIPE, stderr=sp.PIPE, text=True, encoding="cp437").stdout.splitlines()
    substitles= [x for x in out]
    # Check index of Swedish subtitles
    swe_idx = [i for i, x in enumerate(out) if "swe" in x]
    
    if len(swe_idx) >= 1:
        sv_subs = 1
        if os.path.isfile(f"{(os.path.join(savedir, file))[:-4]}.srt") == False:
            # Save subtitle in srt format
            out = sp.run(['ffmpeg','-i', videofile, '-map', f's:{swe_idx[0]}', '-f','srt', f'{savedir}/{file[:-4]}/file.srt'], stdout=sp.PIPE, stderr=sp.PIPE, universal_newlines=True)
            fuse_subtitles(f'{savedir}/{file[:-4]}/file.srt', f'{savedir}/{file[:-4]}/file.srt')
        else:
            pass
        if format == "mp3":
        # Save sound in mp3 format
            out = sp.run(['ffmpeg', '-i' , videofile, '-acodec',  'libmp3lame', f'{savedir}/{file[:-4]}/file.mp3'])
        # Save sound in wav format
        elif format == "wav":   
            out = sp.run(['ffmpeg', '-i' , videofile, '-acodec',  'pcm_s16le', '-ac', '1', '-ar', '16000', f'{savedir}/{file[:-4]}/file.wav'])          
    else:
        sv_subs = 0
        print(f"Swedish subtitles are not available for {file}.")
        
    return(sv_subs, substitles)
        
    
def create_statistics(kanal, file, sv_subs, subs, csv_file):
    """ Create a dataframe with info about available subtitles."""
    
    exists = os.path.exists(csv_file)
    
    with open(csv_file, mode='a', newline='') as save_file:
            fieldnames = ['channel', 'filename', 'swedish_subtitles', 'subtitles']
            writer = csv.DictWriter(save_file, fieldnames=fieldnames)

            # Write the header row
            if not exists:
                writer.writeheader()
            writer.writerow({
                    'channel': kanal,
                    'filename': file[:-4],
                    'swedish_subtitles': sv_subs,
                    'subtitles': subs
                    })
            
def main():

    rootdir = args["i"]
    savedir = args["o"]
    format = args["sound_format"]
    csv = args["csv"]
    
    for subdir, dirs, files in os.walk(rootdir):
        kanal = subdir.split("/")[8:10] #adjust to your path, should point to the channel/subchannel part of the directory (e.g. "cmore/cmorefirst")
        kanal = ("/").join(kanal)

        #filter only channels with Swedish subtitles 
        if kanal in channels:
            for file in files:
                videofile_path = os.path.join(subdir, file)
                if not os.path.exists(os.path.join(savedir, file[:-4])):
                   makedirs(os.path.join(savedir, file[:-4]))
                video_savedir = os.path.join(savedir, file[:-4])
                if os.path.isfile(f"{video_savedir}/file.{format}") == False:
                    sv_subs, subs = check_and_extract(videofile_path, file, savedir, format)
                    create_statistics(kanal, file, sv_subs, subs, csv)
                    
                    if sv_subs == 0:
                        os.rmdir(os.path.join(savedir, file[:-4]))
                else:
                  print(f"Skipping, {file[:-4]}.{format} already exists.")

if __name__ == "__main__":
    start = time.time()
    print(args)
    main()
    print(time.time() - start)

