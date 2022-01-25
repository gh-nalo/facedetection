import pandas as pd
import os
from FilePaths import Base


class SaveTimings:

    def __init__(self, filename, save_results=False):
        self.save = save_results
        self.times = []
        self.filename = filename
        self.path = os.path.join(Base.RESULTS, self.filename) + ".xlsx"
        print(self.path)

    def new_value(self, time) -> int:

        self.times.append(time)

        return len(self.times)

    def save_results(self) -> None:

        if not self.save:
            return

        df = pd.DataFrame(self.times, columns=[f"{self.filename}_FPS"])
        df.to_excel(self.path, index=False)
