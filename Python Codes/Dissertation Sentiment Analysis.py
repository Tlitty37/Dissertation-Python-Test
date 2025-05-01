import pandas as pd
from textblob import TextBlob

excel_file = '/Users/tobylodge/Documents/songs_output.xlsx'  
df = pd.read_excel(excel_file)
df.columns = df.columns.str.strip()
lyrics_file = '/Users/tobylodge/Desktop/LYRICS IMPORTANT/lyrics_combined.txt'  
with open(lyrics_file, 'r', encoding='utf-8') as file:
    lyrics_content = file.read()


songs = lyrics_content.split('--------------------------------------------------')

processed_count = 0

for song in songs:
    if not song.strip():
        continue
    lines = song.strip().split('\n')
    artist = None
    song_name = None
    for line in lines:
        if line.startswith('Artist:'):
            artist = line.replace('Artist:', '').strip()
        elif line.startswith('Song:'):
            song_name = line.replace('Song:', '').strip()
    if not artist or not song_name:
        print(f"Error: Skipping malformed song block (artist or song line missing).")
        continue
    lyrics = '\n'.join(lines[2:]).strip()

    blob = TextBlob(lyrics)
    polarity = blob.sentiment.polarity  
    subjectivity = blob.sentiment.subjectivity  

    match = df[(df['Artist Name'] == artist) & (df['Song Name'] == song_name)]
    if not match.empty:
        index = match.index[0]
        df.at[index, 'I'] = polarity
        df.at[index, 'J'] = subjectivity  
    else:
        print(f"Error: No match found for Artist: {artist}, Song: {song_name}")
    processed_count += 1
    if processed_count % 20 == 0:
        df.to_excel(excel_file, index=False)
        print(f"Progress: Saved after processing {processed_count} songs.")

df.to_excel(excel_file, index=False)
print("FINISHED: Final sentiment analysis saved.")
