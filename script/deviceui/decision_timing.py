"""Receiver-side decision timing; no floor, synthesis or cancellation controls."""
import time

from turn_head import HiddenReplyDecoder


class ReceivedReply(bytes):
    def __new__(cls, data, at, monotonic, serial):
        result = super().__new__(cls, data)
        result.received_at = at
        result.received_monotonic = monotonic
        result.read_serial = serial
        return result


class TimedHiddenReplyDecoder:
    """Timestamp the read containing the last byte, before parsing its frames.

    Multiple tokens may share a transport read, especially with MTP. Equal read
    stamps are intentionally equal, not interpolated into fictitious token times.
    """
    def __init__(self, snapshot):
        self.snapshot = snapshot
        self.serial = 0
        self.stamp = None
        self.decoder = HiddenReplyDecoder(self.on_snapshot)

    def on_snapshot(self, frame):
        at, mono, serial = self.stamp
        frame.update(transport_received_at=at, transport_received_monotonic=mono,
                     transport_read_serial=serial)
        self.snapshot(frame)

    def feed(self, data):
        self.serial += 1
        self.stamp = (time.time(), time.monotonic(), self.serial)
        return [ReceivedReply(part, *self.stamp) for part in self.decoder.feed(data)]

    def finish(self):
        self.decoder.finish()


def decision_stamp(action, source, metadata_before, chunk):
    recognized_at, recognized_mono = time.time(), time.monotonic()
    return dict(action=action, source=source, metadata_before=metadata_before,
        received_at=getattr(chunk, 'received_at', recognized_at),
        received_monotonic=getattr(chunk, 'received_monotonic', recognized_mono),
        read_serial=getattr(chunk, 'read_serial', None),
        recognized_at=recognized_at, recognized_monotonic=recognized_mono,
        receipt_source='transport_read' if hasattr(chunk, 'received_monotonic') else 'parser_fallback')


def compare_decision_timing(head, text, requested_monotonic):
    snap = head['snapshot']
    boundary = snap['transport_received_monotonic']
    return dict(origin='complete_hidden_frame_received', clock='same_process_monotonic',
        text_source=text['source'], receipt_source=text['receipt_source'],
        metadata_before_decision=text['metadata_before'],
        head_action=head['action'], text_action=text['action'],
        head_after_frame_s=head['ready_monotonic']-boundary,
        text_after_frame_s=text['received_monotonic']-boundary,
        parser_after_frame_s=text['recognized_monotonic']-boundary,
        parser_after_receipt_s=text['recognized_monotonic']-text['received_monotonic'],
        text_minus_head_s=text['received_monotonic']-head['ready_monotonic'],
        head_from_request_s=head['ready_monotonic']-requested_monotonic,
        text_from_request_s=text['received_monotonic']-requested_monotonic,
        same_transport_read=text['read_serial']==snap['transport_read_serial'],
        engine_prefill_to_text_wall_s=text['received_at']-snap['snapshot_at'],
        engine_prefill_to_head_wall_s=head['ready_at']-snap['snapshot_at'],
        controlling=False)
