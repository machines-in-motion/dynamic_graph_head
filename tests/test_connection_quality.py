import time
import pytest
from dynamic_graph_head.connection_quality import (
    PacketLossAnalyser,
    PacketLossStatistics,
)


def test_packetlossstatistics_eq():
    x = PacketLossStatistics(1000, 10, 0.001, 123, 12, 0.3)
    almost = PacketLossStatistics(1000, 10, 0.001, 123, 12, 3 * 0.1)
    diff_cs = PacketLossStatistics(1001, 10, 0.001, 123, 12, 0.3)
    diff_cl = PacketLossStatistics(1000, 11, 0.001, 123, 12, 0.3)
    diff_cr = PacketLossStatistics(1000, 10, 0.01, 123, 12, 0.3)
    diff_ss = PacketLossStatistics(1000, 10, 0.001, 323, 12, 0.3)
    diff_sl = PacketLossStatistics(1000, 10, 0.001, 123, 9, 0.3)
    diff_sr = PacketLossStatistics(1000, 10, 0.001, 123, 12, 0.33)

    assert x == x
    assert x == almost
    assert x != diff_cs
    assert x != diff_cl
    assert x != diff_cr
    assert x != diff_ss
    assert x != diff_sl
    assert x != diff_sr


def test_packetlossstaticis_repr_eval():
    x = PacketLossStatistics(1000, 10, 0.001, 123, 12, 0.3)

    # for a simple class like this repr() should return something that can be
    # eval'ed into an equivalent object
    assert x == eval(repr(x))


def test_packet_loss_analyser_analyse():
    pla = PacketLossAnalyser(0.1, 3)

    # report without update should raise error
    with pytest.raises(RuntimeError):
        pla.analyse_total()
    with pytest.raises(RuntimeError):
        pla.analyse_window()

    # add one
    pla.update(1000, 10, 0, 0)
    assert len(pla.buffer) == 1
    assert pla.analyse_total() == PacketLossStatistics(
        1000, 10, 0.01, 0, 0, 0.0
    )
    assert pla.analyse_window() == PacketLossStatistics(0, 0, 0.0, 0, 0, 0.0)

    # make sure enough time has passed, then make another update
    time.sleep(0.11)
    pla.update(1200, 10, 100, 3)
    assert len(pla.buffer) == 2
    assert pla.analyse_total() == PacketLossStatistics(
        1200, 10, 10 / 1200, 100, 3, 3 / 100
    )
    assert pla.analyse_window() == PacketLossStatistics(
        200, 0, 0.0, 100, 3, 3 / 100
    )

    # and another update
    time.sleep(0.11)
    pla.update(1500, 15, 300, 30)
    assert len(pla.buffer) == 3
    assert pla.analyse_total() == PacketLossStatistics(
        1500, 15, 15 / 1500, 300, 30, 30 / 300
    )
    assert pla.analyse_window() == PacketLossStatistics(
        500, 5, 0.01, 300, 30, 0.1
    )

    # the buffer is full now, so with the next update, the first sample should
    # be dropped
    time.sleep(0.11)
    pla.update(2000, 40, 430, 33)
    assert len(pla.buffer) == 3
    assert pla.analyse_total() == PacketLossStatistics(
        2000, 40, 40 / 2000, 430, 33, 33 / 430
    )
    assert pla.analyse_window() == PacketLossStatistics(
        800, 30, 30 / 800, 330, 30, 30 / 330
    )


def test_packet_loss_analyser_fast_update():
    # set a very high update rate for this test
    pla = PacketLossAnalyser(5.0, 3)

    pla.update(1000, 10, 0, 0)
    pla.update(2000, 10, 0, 0)
    pla.update(3000, 10, 0, 0)

    # since update is called within one update period, only the first one
    # should be added
    assert len(pla.buffer) == 1
    assert pla.analyse_total() == PacketLossStatistics(
        1000, 10, 0.01, 0, 0, 0.0
    )
