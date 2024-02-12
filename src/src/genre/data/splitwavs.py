from pydub import AudioSegment
import math

class SplitWavAudio():
    def __init__(self, folder, filename, path):
        self.folder = folder
        self.filename = filename
        self.filepath = path
        
        self.audio = AudioSegment.from_file(self.filepath, format="wav")

    
    def get_duration(self):
        """Measures duration of a SplitWavAudio audio attribute

        Returns
        -------
         int
           duration of audio in seconds
        """
        return self.audio.duration_seconds
    
    def single_split(self, from_min, to_min, split_filename):
        """Exports a clip of audio from a SplitWavAudio object to a designated file

        Parameters
        ----------
        from_min: int
            time at beginning of audio clip
        to_min: int
            time at end of audio clip
        split_filename: string
            name of audio file to export to
        """
        t1 = from_min * 60 * 1000
        t2 = to_min * 60 * 1000
        split_audio = self.audio[t1:t2]
        split_audio.export(self.folder + '\\' + split_filename, format="wav")
        
    def multiple_split(self, min_per_split):
        """Splits a SplitWavAudio object into segments of equal length and exports them to a folder.
        The last segment of audio will be shorter than the desired length because it will stop when the song ends.

        Parameters
        ----------
        min_per_split: int
            length of audio clip in minutes
        """
        total_mins = math.ceil(self.get_duration() / 60)
        for i in range(0, total_mins, min_per_split):
            split_fn = str(int(i/min_per_split)) + '_' + self.filename # Filenames will be the number of the split, then the audio filename. ex: the second 3 minute clip of "audio.wav" is "2_audio.wav"
            self.single_split(i, i+min_per_split, split_fn)
            print(str(split_fn) + ' Done')
            if i == total_mins - min_per_split:
                print('All split successfully')