#!/usr/bin/env python3


EYE_TRACKING_VIDEO_LIST_STR = [
  "BNS",  # or "JNG"
  "BRZ",
  "DST",
  "EXR",
  "JNG",  # or "BNS"
  "PRS",
  "BOT",
  "RFS",
  "RPW",
  "RST",
  "TRT",
  "ZMZ",
]

ECG_GSR_VIDEO_LIST_STR = [
"Baseline",
"BNS", # or JNG
"PRS",
"BRZ",
"RST",
"DST",
"EXR",
"JNG",  # or BNS
"BOT",
"RFS",
"RPW",
"TNT",
"ZMZ",
]


def get_str_code(index: int, data_type: str) -> str:
    """Returns the Str_Code"""
    eye_tracking_types = ("eye_tracking", "eye")
    ecg_gsr_types = ("ecg_gsr", "ecg", "gsr")
    if data_type in eye_tracking_types:
        if index > len(EYE_TRACKING_VIDEO_LIST_STR):
            raise ValueError(f"Index too large ({index})")
        return EYE_TRACKING_VIDEO_LIST_STR[index]
    elif data_type in ecg_gsr_types:
        if index > len(ECG_GSR_VIDEO_LIST_STR):
            raise ValueError(f"Index too large ({index})")
        return ECG_GSR_VIDEO_LIST_STR[index]
    else:
        raise ValueError(f"Invalid data_type: {data_type}. Allowed values: "
                         f"{", ".join(eye_tracking_types)}, {", ".join(ecg_gsr_types)}")