import PySimpleGUI as gui

from modules import constants, event_processor


def app_gui() -> object:
    """
    Building GUI for the app.

    Returns:
        object: PysimpleGUI object
    """
    gui_font = ("Arial", 40)
    centered_column = [
        [gui.Text(constants.WINDOW_NAME, font=gui_font)],
        [gui.Push()],
        [gui.Button(constants.START_BUTTON_TXT), gui.Button(constants.STOP_BUTTON_TXT)],
    ]
    gui_layout = [
        [gui.VPush()],
        [
            gui.Push(),
            gui.Column(centered_column, element_justification="c"),
            gui.Push(),
        ],
        [gui.VPush()],
    ]

    gui_window = gui.Window(
        constants.WINDOW_NAME, gui_layout, size=(constants.WINDOW_WIDTH, constants.WINDOW_HEIGHT)
    )
    return gui_window


def the_thread():
    """
    This function runs our process on an different thread to avoid the GUI from freezing
    """
    while True:
        event_processor.event_handler()


def main() -> None:
    """
    This is entrypoint for the app.
    """
    gui_window = app_gui()

    while True:
        event, _ = gui_window.read()
        if event == gui.WIN_CLOSED or event == constants.STOP_BUTTON_TXT:
            break

        if event == constants.START_BUTTON_TXT:
            print("App started succcessfully")
            gui_window.start_thread(lambda: the_thread(), "-THREAD FINISHED-")

    gui_window.close()


if __name__ == "__main__":
    main()
