"""An original primitive-based map; no game art or live engine objects."""

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen, QPolygonF
from PySide6.QtWidgets import QSizePolicy, QWidget

from desktop.i18n import bind_text, language_manager, register_translations, tr

register_translations(
    {
        "Stage map": "关卡地图",
        "Run or inspect a stage to show its map": "运行或查看关卡后显示地图",
        "IN": "入口",
        "GOAL": "目标",
        "Blue: operator   Green: summon   Red: enemy   Dashed: route   Gold: block": "蓝色：干员   绿色：召唤物   红色：敌人   虚线：路线   金色：阻挡",
        "Tile {position}\n{key}\nDeployment: {buildable}\nPassable: {passable}": "格子 {position}\n{key}\n可部署：{buildable}\n可通行：{passable}",
        "MELEE": "近战",
        "RANGED": "远程",
        "ALL": "全部",
        "NONE": "无",
    }
)


class StageMap(QWidget):
    tile_selected = Signal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._snapshot = None
        self._map_rect = QRectF()
        self._cell = 1.0
        self._hover_tile = None
        self.setMinimumSize(440, 240)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)
        bind_text(self, "Stage map", "setAccessibleName")
        language_manager.language_changed.connect(self._retranslate)

    def set_snapshot(self, snapshot):
        self._snapshot = snapshot
        self.update()

    def _point(self, position):
        return QPointF(
            self._map_rect.left() + (position[0] + 0.5) * self._cell,
            self._map_rect.bottom() - (position[1] + 0.5) * self._cell,
        )

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), self.palette().base())
        data = self._snapshot
        if data is None or not data.tiles:
            painter.setPen(self.palette().text().color())
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                tr("Run or inspect a stage to show its map"),
            )
            return
        self._cell = min(
            (self.width() - 64) / data.width, (self.height() - 70) / data.height
        )
        width, height = self._cell * data.width, self._cell * data.height
        self._map_rect = QRectF(
            (self.width() - width) / 2, (self.height() - height - 25) / 2, width, height
        )
        dark = self.palette().base().color().lightness() < 128
        ground = QColor("#34434c" if dark else "#e4eaee")
        high = QColor("#526675" if dark else "#c2d1dc")
        blocked = QColor("#1d252b" if dark else "#8d99a3")
        for y, row in enumerate(data.tiles):
            for x, tile in enumerate(row):
                center = self._point((x, y))
                rect = QRectF(
                    center.x() - self._cell / 2 + 1,
                    center.y() - self._cell / 2 + 1,
                    self._cell - 2,
                    self._cell - 2,
                )
                color = (
                    high
                    if tile.buildable == "RANGED"
                    else blocked
                    if tile.buildable == "NONE"
                    else ground
                )
                if tile.key == "tile_start":
                    color = QColor("#963c4a" if dark else "#f2c0c6")
                elif tile.key == "tile_end":
                    color = QColor("#285e96" if dark else "#b6d6f7")
                painter.fillRect(rect, color)
                painter.setPen(QColor("#aab8c3" if dark else "#4e5a64"))
                if tile.key in ("tile_start", "tile_end"):
                    painter.drawText(
                        rect,
                        Qt.AlignmentFlag.AlignCenter,
                        tr("IN") if tile.key == "tile_start" else tr("GOAL"),
                    )
        painter.setPen(QPen(QColor(229, 170, 71, 160), 1.6, Qt.PenStyle.DashLine))
        for route in data.routes:
            if not route:
                continue
            path = QPainterPath(self._point(route[0]))
            for point in route[1:]:
                path.lineTo(self._point(point))
            painter.drawPath(path)
        units = {unit.id: unit for unit in data.units}
        painter.setPen(QPen(QColor("#f0b94e"), 2))
        for unit in data.units:
            if unit.blocked_by in units:
                painter.drawLine(
                    self._point(unit.position),
                    self._point(units[unit.blocked_by].position),
                )
        radius = min(14, self._cell * 0.22)
        for unit in data.units:
            center = self._point(unit.position)
            painter.setPen(QPen(QColor("#f0f4f6"), 1.3))
            painter.setBrush(
                QColor("#7ecba1" if unit.kind == "summon" else "#e8b851" if unit.kind == "device" else "#57a8eb" if unit.kind == "operator" else "#e46369")
            )
            if unit.kind in ("operator","device","summon"):
                painter.drawRoundedRect(
                    QRectF(
                        center.x() - radius, center.y() - radius, radius * 2, radius * 2
                    ),
                    3,
                    3,
                )
                dx, dy = ((1, 0), (0, -1), (-1, 0), (0, 1))[unit.direction]
                tip = QPointF(
                    center.x() + dx * radius * 1.8, center.y() + dy * radius * 1.8
                )
                side = radius * 0.4
                triangle = QPolygonF(
                    [
                        tip,
                        QPointF(
                            center.x() + dx * radius - dy * side,
                            center.y() + dy * radius + dx * side,
                        ),
                        QPointF(
                            center.x() + dx * radius + dy * side,
                            center.y() + dy * radius - dx * side,
                        ),
                    ]
                )
                painter.drawPolygon(triangle)
                painter.setPen(self.palette().text().color())
                painter.drawText(
                    QRectF(
                        center.x() - self._cell / 2,
                        center.y() + radius + 5,
                        self._cell,
                        18,
                    ),
                    Qt.AlignmentFlag.AlignCenter,
                    unit.name,
                )
            else:
                painter.drawEllipse(center, radius * 0.68, radius * 0.68)
            bar = QRectF(center.x() - radius, center.y() - radius - 7, radius * 2, 3)
            painter.fillRect(bar, QColor("#273039"))
            bar.setWidth(radius * 2 * max(0, min(1, unit.hp / max(1, unit.max_hp))))
            painter.fillRect(bar, QColor("#61cba3"))
        painter.setPen(self.palette().text().color())
        for x in range(data.width):
            point = self._point((x, 0))
            painter.drawText(
                QRectF(point.x() - 10, self._map_rect.bottom() + 3, 20, 18),
                Qt.AlignmentFlag.AlignCenter,
                str(x),
            )
        for y in range(data.height):
            point = self._point((0, y))
            painter.drawText(
                QRectF(self._map_rect.left() - 23, point.y() - 9, 20, 18),
                Qt.AlignmentFlag.AlignCenter,
                str(y),
            )
        painter.drawText(
            QRectF(12, self.height() - 24, self.width() - 24, 20),
            Qt.AlignmentFlag.AlignCenter,
            tr(
                "Blue: operator   Green: summon   Red: enemy   Dashed: route   Gold: block"
            ),
        )

    def _tile_at(self, point):
        if self._snapshot is None or not self._map_rect.contains(point):
            return None
        x = int((point.x() - self._map_rect.left()) / self._cell)
        y = (
            self._snapshot.height
            - 1
            - int((point.y() - self._map_rect.top()) / self._cell)
        )
        return x, y

    def mousePressEvent(self, event):
        tile = self._tile_at(event.position())
        if tile is not None and event.button() == Qt.MouseButton.LeftButton:
            self.tile_selected.emit(*tile)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        position = self._tile_at(event.position())
        self._hover_tile = position
        self._refresh_tooltip()
        super().mouseMoveEvent(event)

    def _refresh_tooltip(self):
        position = self._hover_tile
        if (
            position is not None
            and self._snapshot is not None
            and 0 <= position[0] < self._snapshot.width
            and 0 <= position[1] < self._snapshot.height
        ):
            tile = self._snapshot.tiles[position[1]][position[0]]
            self.setToolTip(
                tr(
                    "Tile {position}\n{key}\nDeployment: {buildable}\nPassable: {passable}"
                ).format(
                    position=position,
                    key=tile.key,
                    buildable=tr(tile.buildable),
                    passable=tr(tile.passable),
                )
            )
        else:
            self.setToolTip("")

    def _retranslate(self, *_):
        self._refresh_tooltip()
        self.update()
