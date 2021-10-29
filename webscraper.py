from lyricsgenius import Genius

def get_lyrics(artist, n_songs=None, format='list'):
    '''
    Retrieves lyrics for artist and returns in specified format.
    '''
    genius = Genius('E-oQIDZY3DrYoMrYZsACfwP28y6FSVTC2E3ir_9caMHZB4oSn7bScpW5iC0t04a4', remove_section_headers=True)
    songs = genius.search_artist(artist, max_songs=n_songs).songs
    
    if format == 'list':
        lyrics = [song.lyrics for song in songs]

    return lyrics

