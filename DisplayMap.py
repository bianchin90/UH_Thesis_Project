import sys
import io
import folium  # pip install folium
from PyQt5.QtWidgets import QApplication, QWidget, QHBoxLayout, QVBoxLayout
from PyQt5.QtWebEngineWidgets import QWebEngineView  # pip install PyQtWebEngine
import time

"""
Folium in PyQt5
"""


class MyApp(QWidget):

    def __init__(self):
        super().__init__()
        self.setWindowTitle('Folium in PyQt Example')
        self.window_width, self.window_height = 1200, 900
        self.setMinimumSize(self.window_width, self.window_height)

        layout = QVBoxLayout()
        self.setLayout(layout)

        coordinate = (42.64018745751431, 12.570672933933098)
        m = folium.Map(
            tiles='Stamen Terrain',
            zoom_start=6,
            location=coordinate
        )

        folium.Marker(
            location=[44.97425925588055, 9.276561443043326],
            popup="Lombardia",
            icon=folium.Icon(icon="cloud"),
        ).add_to(m)

        folium.Marker(
            location=[40.85274233825337, 14.315845878353649],
            popup="Campania",
            icon=folium.Icon(color="green"),
        ).add_to(m)

        folium.Marker(
            location=[38.26288559444078, 13.377312456164717],
            popup="Some Other Location",
            icon=folium.Icon(color="red", icon="info-sign"),
        ).add_to(m)

        # save map data to data object
        data = io.BytesIO()
        m.save(data, close_file=False)

        webView = QWebEngineView()
        webView.setHtml(data.getvalue().decode())
        layout.addWidget(webView)

    def add_points(self):
        folium.Marker(
            location=[44.97425925588055, 9.276561443043326],
            popup="Lombardia",
            icon=folium.Icon(icon="cloud"),
        ).add_to(self.m)
        time.sleep(5)

        folium.Marker(
            location=[40.85274233825337, 14.315845878353649],
            popup="Campania",
            icon=folium.Icon(color="green"),
        ).add_to(self.m)
        time.sleep(5)

        folium.Marker(
            location=[38.26288559444078, 13.377312456164717],
            popup="Some Other Location",
            icon=folium.Icon(color="red", icon="info-sign"),
        ).add_to(self.m)
        time.sleep(5)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyleSheet('''
        QWidget {
            font-size: 35px;
        }
    ''')

    myApp = MyApp()
    myApp.show()

    print('Adding markers..')
    #myApp.add_points()

    try:
        sys.exit(app.exec_())
    except SystemExit:
        print('Closing Window...')
