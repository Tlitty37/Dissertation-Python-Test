import requests
from bs4 import BeautifulSoup
import pandas as pd
genre_page_url = 'https://getsongkey.com/genres'
response = requests.get(genre_page_url)
soup = BeautifulSoup(response.text, 'html.parser')

song_data = []
seen_songs = set()

genre_counters = {}
artist_counters = {}
total_song_count = 0

genres = soup.find_all('a', href=True)

for genre in genres:
    try:
        genre_link = genre['href']
        genre_name = genre.find('h4', itemprop='name').text.strip()
        if genre_name not in genre_counters:
            genre_counters[genre_name] = {"Major": 0, "Minor": 0}
        if genre_counters[genre_name]["Major"] >= 500 and genre_counters[genre_name]["Minor"] >= 500:
            print(f"Skipping genre: {genre_name} (limits reached)")
            continue
        full_genre_url = f'https://getsongkey.com{genre_link}'
        genre_response = requests.get(full_genre_url)
        genre_soup = BeautifulSoup(genre_response.text, 'html.parser')
        artists = genre_soup.find_all('div', class_='co-xs-6 col-sm-3')
        for artist in artists:
            try:
                artist_link = artist.find('a', itemprop='url')['href']
                artist_name = artist.find('h4', itemprop='name').text.strip()
                if artist_name not in artist_counters:
                    artist_counters[artist_name] = {"Major": 0, "Minor": 0}
                if artist_counters[artist_name]["Major"] >= 20 and artist_counters[artist_name]["Minor"] >= 20:
                    continue
                full_artist_url = f'https://getsongkey.com{artist_link}'
                artist_response = requests.get(full_artist_url)
                artist_soup = BeautifulSoup(artist_response.text, 'html.parser')
                songs = artist_soup.find_all('li')
                for song in songs:
                    try:
                        song_name = song.find('h5', itemprop='name').text.strip()

                        if (genre_name, artist_name, song_name) in seen_songs:
                            continue
                        key = song.find('div', class_='popular key col-xs-12').text.replace('Key of', '').strip()
                        mode = 'Minor' if 'm' in key else 'Major'
                        if mode == 'Major' and (artist_counters[artist_name]["Major"] >= 20 or genre_counters[genre_name]["Major"] >= 500):
                            continue
                        if mode == 'Minor' and (artist_counters[artist_name]["Minor"] >= 20 or genre_counters[genre_name]["Minor"] >= 500):
                            continue
                        seen_songs.add((genre_name, artist_name, song_name))
                        artist_counters[artist_name][mode] += 1
                        genre_counters[genre_name][mode] += 1
                        total_song_count += 1
                        song_data.append([genre_name, artist_name, song_name, key, mode])
                        if total_song_count % 20 == 0:
                            print(f"Total Songs Added: {total_song_count}")

                    except AttributeError:
                        continue
                if artist_counters[artist_name]["Major"] < 20 or artist_counters[artist_name]["Minor"] < 20:
                    albums = artist_soup.find_all('div', itemprop='album')
                    for album in albums:
                        try:
                            album_link = album.find('a', itemprop='url')['href']
                            full_album_url = f'https://getsongkey.com{album_link}'
                            album_response = requests.get(full_album_url)
                            album_soup = BeautifulSoup(album_response.text, 'html.parser')
                            album_songs = album_soup.find_all('li')
                            for album_song in album_songs:
                                try:
                                    song_name = album_song.find('h5', itemprop='name').text.strip()
                                    if (genre_name, artist_name, song_name) in seen_songs:
                                        continue

                                    key = album_song.find('div', class_='popular key col-xs-12').text.replace('Key of', '').strip()
                                    mode = 'Minor' if 'm' in key else 'Major'

                                    if mode == 'Major' and (artist_counters[artist_name]["Major"] >= 20 or genre_counters[genre_name]["Major"] >= 500):
                                        continue
                                    if mode == 'Minor' and (artist_counters[artist_name]["Minor"] >= 20 or genre_counters[genre_name]["Minor"] >= 500):
                                        continue
                                    seen_songs.add((genre_name, artist_name, song_name))

                                    artist_counters[artist_name][mode] += 1
                                    genre_counters[genre_name][mode] += 1
                                    total_song_count += 1

                                    song_data.append([genre_name, artist_name, song_name, key, mode])
                                    if total_song_count % 20 == 0:
                                        print(f"Total Songs Added: {total_song_count}")

                                except AttributeError:
                                    continue

                        except (AttributeError, TypeError):
                            continue

            except (AttributeError, TypeError):
                continue

    except (AttributeError, TypeError):
        continue

df = pd.DataFrame(song_data, columns=['Genre', 'Artist Name', 'Song Name', 'Key', 'Mode'])
df.to_excel('songkwgenre.xlsx', index=False)

print("Data saved to 'songkwgenre.xlsx'.")
