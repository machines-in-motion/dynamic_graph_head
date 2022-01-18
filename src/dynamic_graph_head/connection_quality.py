"""Utilities for analysing the robot connection quality."""
# FIXME: Probably  better to add this module to the "solo" package?
import collections
import math
import textwrap
import time
import typing


class PacketLossStatistics(typing.NamedTuple):
    """Statistics about sent/lost command and sensor packets."""

    #: Number of command packets sent to the robot.
    commands_sent: int
    #: Number of command packets that were not received by the robot.
    commands_lost: int
    #: Ratio of lost/sent command packets.
    commands_ratio: float

    #: Number of sensor packets sent by the robot.
    sensor_sent: int
    #: Number of sensor packets that were not received.
    sensor_lost: int
    #: Ratio of lost/sent sensor packets.
    sensor_ratio: float

    def __str__(self):
        return (
            "Lost Command Msgs.: {:.1f}% ({:.0f}/{:.0f})\n"
            "Lost Sensor Msgs.: {:.1f}% ({:.0f}/{:.0f})\n"
        ).format(
            self.commands_ratio * 100,
            self.commands_lost,
            self.commands_sent,
            self.sensor_ratio * 100,
            self.sensor_lost,
            self.sensor_sent,
        )

    def __eq__(self, o):
        if not isinstance(o, PacketLossStatistics):
            return NotImplemented

        return (
            self.commands_sent == o.commands_sent
            and self.commands_lost == o.commands_lost
            and math.isclose(self.commands_ratio, o.commands_ratio)
            and self.sensor_sent == o.sensor_sent
            and self.sensor_lost == o.sensor_lost
            and math.isclose(self.sensor_ratio, o.sensor_ratio)
        )


class PacketLossAnalyser:
    """Analyse the amount of lost packets over a fixed time window.

    The lower-level interface of the robot only provides information about the
    total number of sent/lost messages in the communication with the robot,
    i.e. the numbers are accumulated over the whole runtime of the robot.
    This class analyses regular updates of this data to get information about
    the recent connection quality.
    """

    def __init__(self, update_rate_s: float, window_size: int):
        """
        Args:
            update_rate_s: Rate (in seconds) at which updates should be added.
            window_size: Number of previous updates that are kept in the
                buffer.  The window size in seconds can be computed as
                ``window_size * update_rate_s``.
        """
        self.update_rate_s = update_rate_s
        self.buffer: typing.Deque = collections.deque(maxlen=window_size)
        self._last_update = 0.0

    def update(
        self,
        sent_command: int,
        lost_command: int,
        sent_sensor: int,
        lost_sensor: int,
    ) -> bool:
        """Check if next update is due and, if yes, update the buffer.

        Args:
            sent_command: Total number of sent command messages.
            lost_command: Total number of lost command messages.
            sent_sensor: Total number of sent sensor messages.
            lost_sensor: Total number of lost sensor messages.

        Returns:
            True if an update is performed, false if not.
        """
        now = time.time()
        if now - self._last_update > self.update_rate_s:
            self.buffer.append(
                (sent_command, lost_command, sent_sensor, lost_sensor)
            )
            self._last_update = now

            return True

        return False

    def analyse_total(self) -> PacketLossStatistics:
        """Get total number of packets sent/lost.

        This covers the complete runtime of the robot.
        """
        if not self.buffer:
            raise RuntimeError("Not data to analyse.  Call `update()` first.")

        newest = self.buffer[-1]
        return self._compute_stats((0, 0, 0, 0), newest)

    def analyse_window(self) -> PacketLossStatistics:
        """Get number of packets sent/lost in the current time window."""
        if not self.buffer:
            raise RuntimeError("Not data to analyse.  Call `update()` first.")

        oldest = self.buffer[0]
        newest = self.buffer[-1]

        return self._compute_stats(oldest, newest)

    def __str__(self):
        stats_total = self.analyse_total()
        stats_buffer = self.analyse_window()
        window_size_s = self.update_rate_s * len(self.buffer)

        report = "".join(
            (
                "-----------------\n",
                "Total:\n",
                textwrap.indent(str(stats_total), "\t"),
                "Window ({}s):\n".format(window_size_s),
                textwrap.indent(str(stats_buffer), "\t"),
            )
        )

        return report

    def _compute_stats(self, start, end):
        """Compute stats for the given start and end values."""
        diff = (
            end[0] - start[0],
            end[1] - start[1],
            end[2] - start[2],
            end[3] - start[3],
        )

        def ratio(lost, sent):
            return lost / sent if sent else 0.0

        return PacketLossStatistics(
            diff[0],
            diff[1],
            ratio(diff[1], diff[0]),
            diff[2],
            diff[3],
            ratio(diff[3], diff[2]),
        )
