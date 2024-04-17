from window import Window


class Main:
    def __init__(self) -> None:
        self.win = Window("mimikyu.jpeg", 64, 64, 5)

    def running(self) -> None:
        self.win.running()


if __name__ == "__main__":
    main = Main()
    main.running()
