package com.example.benchmarking

import android.os.Build

object DeviceInfo {
    fun getDeviceName(): String {
        return "${Build.MANUFACTURER} ${Build.MODEL} (${Build.HARDWARE})"
    }
}
