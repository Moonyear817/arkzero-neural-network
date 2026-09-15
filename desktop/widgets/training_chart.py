"""Bounded QtCharts series; updates are batched by Dashboard's timer."""

from math import ceil, floor

from PySide6.QtCharts import QChart, QChartView, QLineSeries, QValueAxis
from PySide6.QtCore import QEvent, QMargins, Qt
from PySide6.QtGui import QPalette

from desktop.i18n import language_manager, tr


class TrainingChart(QChartView):
    def __init__(self, title, parent=None):
        chart = QChart()
        super().__init__(chart, parent)
        chart.setTitle(title + " · waiting for metrics")
        chart.legend().hide()
        chart.setAnimationOptions(QChart.AnimationOption.NoAnimation)
        chart.setBackgroundVisible(False)
        chart.setMargins(QMargins(12, 4, 8, 4))
        chart.layout().setContentsMargins(0, 0, 0, 0)
        self.title = title
        language_manager.language_changed.connect(self.retranslate)
        self.series = QLineSeries()
        chart.addSeries(self.series)
        self.x_axis, self.y_axis = QValueAxis(), QValueAxis()
        # Qt's default fractional labels crowd a short iteration axis on macOS.
        self.x_axis.setLabelFormat("%.0f")
        self.x_axis.setTickCount(3)
        self.y_axis.setLabelFormat("%.3g")
        self.y_axis.setTickCount(3)
        for axis in (self.x_axis, self.y_axis):
            axis.setTruncateLabels(False)
            font = axis.labelsFont()
            font.setPointSizeF(10)
            axis.setLabelsFont(font)
            axis.setTitleFont(font)
        self.x_axis.setTitleText("Iteration / sample")
        chart.addAxis(self.x_axis, Qt.AlignmentFlag.AlignBottom)
        chart.addAxis(self.y_axis, Qt.AlignmentFlag.AlignLeft)
        self.series.attachAxis(self.x_axis)
        self.series.attachAxis(self.y_axis)
        self.series.setMarkerSize(7)
        self.series.pointsRemoved.connect(self.retranslate)
        self.setMinimumSize(290, 180)
        self.retranslate()
        self.update_theme()

    def update_theme(self):
        dark = self.palette().color(QPalette.ColorRole.Window).lightness() < 128
        self.chart().setTheme(
            QChart.ChartTheme.ChartThemeDark
            if dark
            else QChart.ChartTheme.ChartThemeLight
        )
        self.chart().setBackgroundVisible(False)

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() == QEvent.Type.PaletteChange and hasattr(self, "series"):
            self.update_theme()

    def retranslate(self, *_):
        self.chart().setTitle(
            tr(self.title)
            + (" · " + tr("waiting for metrics") if not self.series.count() else "")
        )
        self.x_axis.setTitleText(tr("Iteration / sample"))
        self.x_axis.setVisible(bool(self.series.count()))
        self.y_axis.setVisible(bool(self.series.count()))

    def add_value(self, x, y):
        self.chart().setTitle(tr(self.title))
        self.x_axis.setVisible(True)
        self.y_axis.setVisible(True)
        self.series.append(float(x), float(y))
        if self.series.count() > 300:
            self.series.removePoints(0, self.series.count() - 300)
        points = self.series.points()
        self.series.setPointsVisible(len(points) == 1)
        lower = floor(min(point.x() for point in points))
        upper = ceil(max(point.x() for point in points))
        if lower == upper:
            lower, upper = max(0, lower - 1), upper + 1
        # Three integer ticks; a single sample gets room for its visible marker.
        step = max(1, ceil((upper - lower) / 2))
        self.x_axis.setRange(lower, lower + step * 2)
        values = [p.y() for p in points]
        margin = max(0.01, (max(values) - min(values)) * 0.1)
        self.y_axis.setRange(min(values) - margin, max(values) + margin)
