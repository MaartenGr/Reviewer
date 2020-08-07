class MovieNotFoundError(Exception):
    def __init__(self, input_movie, movies):
        self.input_movie = input_movie
        self.movies = movies

    def __str__(self):
        message = f"{self.input_movie} is not a saved movie. Please select one of the following:"
        for movie in self.movies:
            message += f"\n * {movie}"
        return message