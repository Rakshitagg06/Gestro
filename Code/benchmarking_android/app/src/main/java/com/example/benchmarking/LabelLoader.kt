package com.example.benchmarking

import android.content.Context

object LabelLoader {
    fun loadLabels(context: Context, filename: String = "labels.txt"): List<String> {
        return context.assets.open(filename).bufferedReader().useLines { lines ->
            lines.filter { it.isNotBlank() }.map { it.trim() }.toList()
        }
    }
}
