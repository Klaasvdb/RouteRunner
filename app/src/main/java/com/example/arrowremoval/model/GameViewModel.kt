package com.example.arrowremoval.model

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

data class GameUiState(
    val maze: MazeState = TestLevel.build(),
    val animation: SlitherAnimation? = null,
    val isComplete: Boolean = false,
    val invalidTapPos: GridPos? = null  // briefly highlights an invalid tap
)

class GameViewModel : ViewModel() {

    private val _uiState = MutableStateFlow(GameUiState())
    val uiState: StateFlow<GameUiState> = _uiState.asStateFlow()

    private val originalMaze = TestLevel.build()

    fun onCellTapped(row: Int, col: Int) {
        val state = _uiState.value
        if (state.animation != null) return  // ignore taps during animation

        val maze = state.maze
        val cell = maze.cell(row, col)
        if (cell.isWall || cell.arrow == null || cell.isRemoved) return

        if (!maze.canRemove(row, col)) {
            // Flash the cell to signal it's blocked
            _uiState.value = state.copy(invalidTapPos = GridPos(row, col))
            viewModelScope.launch {
                delay(400)
                _uiState.value = _uiState.value.copy(invalidTapPos = null)
            }
            return
        }

        val anim = maze.buildSlither(row, col) ?: return
        val newMaze = maze.withRemoved(row, col)

        _uiState.value = state.copy(animation = anim, maze = newMaze)

        viewModelScope.launch {
            animateSlither(anim, newMaze)
        }
    }

    private suspend fun animateSlither(anim: SlitherAnimation, maze: MazeState) {
        val totalSteps = 30
        for (step in 0..totalSteps) {
            val progress = step / totalSteps.toFloat()
            _uiState.value = _uiState.value.copy(
                animation = anim.copy(progress = progress)
            )
            delay(16L)  // ~60fps
        }
        val isComplete = maze.isComplete()
        _uiState.value = _uiState.value.copy(
            animation = null,
            isComplete = isComplete
        )
    }

    fun reset() {
        _uiState.value = GameUiState(maze = originalMaze)
    }
}
