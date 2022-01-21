import pandas as pd
import os

BASE_PATH = "./TimingResults/"


class SaveTimings:

    def __init__(self, filename):

        self.times = []
        self.path = os.path.join(BASE_PATH, filename) + ".xlsx"
        print(self.path)

    def new_value(self, time) -> int:

        self.times.append(time)

        return len(self.times)

    def save_results(self) -> None:

        df = pd.DataFrame(self.times)
        df.to_excel(self.path, index=False)
