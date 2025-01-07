// src/lib.rs

// Bring your other modules into scope
// pub mod envs;
mod grid;
mod envs;

// (Optional) Re-export the FFI functions if you want them accessible via `my_project::...`
pub use grid::{
    create_grid_env,
    destroy_grid_env,
    grid_env_display,
    grid_env_step,
    grid_env_score,
    grid_env_is_game_over,
    grid_env_reset,
};
