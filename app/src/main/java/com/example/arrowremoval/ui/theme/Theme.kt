package com.example.arrowremoval.ui.theme

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color

private val DarkColorScheme = darkColorScheme(
    primary = Color(0xFF5080ff),
    background = Color(0xFF1a1a2e),
    surface = Color(0xFF1a1a2e),
    onBackground = Color(0xFFa0a0d0),
    onSurface = Color(0xFFa0a0d0)
)

@Composable
fun ArrowRemovalTheme(content: @Composable () -> Unit) {
    MaterialTheme(
        colorScheme = DarkColorScheme,
        content = content
    )
}
