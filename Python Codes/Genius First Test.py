import re
import pandas as pd
import lyricsgenius
from difflib import SequenceMatcher


client_access_token = "access-token"  #I don't think I can legally share my access token, so I have removed it from the code
genius = lyricsgenius.Genius(client_access_token)


genius.retries = 0  


file_path = "/Users/tobylodge/Documents/songs_output.xlsx"

try:
    df = pd.read_excel(file_path)
except FileNotFoundError:
    print(f"File not found at {file_path}")
    exit()

def clean_lyrics(lyrics):
    
    lyrics = re.sub(r".*Contributors.*?Lyrics", "", lyrics, flags=re.DOTALL)  
    lyrics = re.sub(r"\[Produced by.*?]", "", lyrics, flags=re.IGNORECASE)  
    lyrics = re.sub(r"\[Video directed by.*?]", "", lyrics, flags=re.IGNORECASE)  
    lyrics = re.sub(r"\d+$", "", lyrics, flags=re.MULTILINE)  
    lyrics = re.sub(r"Embed.*$", "", lyrics, flags=re.MULTILINE).strip()  
    return lyrics.strip()

def remove_duplicate_chunks(lyrics):
    lyrics = "\n".join([line.strip() for line in lyrics.split("\n") if line.strip()])
    lines = lyrics.split("\n")
    midpoint = len(lines) // 2
    first_half = "\n".join(lines[:midpoint])
    second_half = "\n".join(lines[midpoint:])
    similarity = SequenceMatcher(None, first_half.strip(), second_half.strip()).ratio()
    if similarity > 0.8:  
        return first_half.strip()  

    return lyrics 

output_file = "lyrics_output_cleaned_full_v3.txt"
no_lyrics_file = "no_lyrics_found.txt"

no_lyrics_list = []


start_row = 2943  # Start from row 869 (0-based index is 867)
num_songs = 323  # Process 500 songs, 3268 is the last one

df_slice = df.iloc[start_row:start_row + num_songs]
with open(output_file, "w", encoding="utf-8") as f, open(no_lyrics_file, "w", encoding="utf-8") as nf:
    for i, row in df_slice.iterrows():
        artist_name = row['Artist Name']
        song_name = row['Song Name']
        try:
            print(f"Searching for \"{song_name}\" by {artist_name}...")
            song = genius.search_song(song_name, artist_name)
            if song:
                lyrics = song.lyrics
                print(f"Lyrics found for {artist_name} - {song_name}")
                lyrics = clean_lyrics(lyrics) 
                lyrics = remove_duplicate_chunks(lyrics) 
                print("Done.")
            else:
                lyrics = "NO LYRICS FOUND"
                no_lyrics_list.append(f"{artist_name} - {song_name}")
                print("Lyrics not found.")
        except Exception as e:
            print(f"Error fetching lyrics for {artist_name} - {song_name}: {e}")
            lyrics = "NO LYRICS FOUND"
            no_lyrics_list.append(f"{artist_name} - {song_name}")
            continue
        f.write(f"Artist: {artist_name}\nSong: {song_name}\n\n")
        f.write(lyrics)
        f.write("\n" + "-" * 50 + "\n")
    nf.write("Songs with no lyrics found:\n")
    for entry in no_lyrics_list:
        nf.write(entry + "\n")

print(f"Lyrics saved to {output_file}")
print(f"Songs with no lyrics found saved to {no_lyrics_file}")
