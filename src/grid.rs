// src/ffi.rs

use crate::envs::grid_world_env::GridEnv;
use crate::envs::basic_env::Env;

// For raw C types:
use std::os::raw::c_void;
// We define the FFI functions here:

#[no_mangle]
pub extern "C" fn create_grid_env() -> *mut GridEnv {
    let rows = 3;
    let cols = 3;
    let s_vec = vec![0; rows * cols];
    let a_vec = vec![0, 1, 2, 3];
    let r_vec = vec![0; rows * cols];
    let t_vec = vec![0, rows * cols - 1];

    let env = GridEnv::new(rows, cols, s_vec, a_vec, r_vec, t_vec);

    // Convert to raw pointer
    Box::into_raw(Box::new(env))
}

#[no_mangle]
pub extern "C" fn destroy_grid_env(env_ptr: *mut GridEnv) {
    if !env_ptr.is_null() {
        unsafe {
            let _ = Box::from_raw(env_ptr); // Drops the Box, freeing the memory
        }
    }
}

#[no_mangle]
pub extern "C" fn grid_env_display(env_ptr: *mut GridEnv) {
    let env = unsafe {
        assert!(!env_ptr.is_null());
        &mut *env_ptr
    };
    env.display();
}

#[no_mangle]
pub extern "C" fn grid_env_step(env_ptr: *mut GridEnv, action: i32) {
    let env = unsafe {
        assert!(!env_ptr.is_null());
        &mut *env_ptr
    };
    env.step(action);
}

#[no_mangle]
pub extern "C" fn grid_env_score(env_ptr: *mut GridEnv) -> f32 {
    let env = unsafe {
        assert!(!env_ptr.is_null());
        &mut *env_ptr
    };
    env.score()
}

#[no_mangle]
pub extern "C" fn grid_env_is_game_over(env_ptr: *mut GridEnv) -> bool {
    let env = unsafe {
        assert!(!env_ptr.is_null());
        &mut *env_ptr
    };
    env.is_game_over()
}

#[no_mangle]
pub extern "C" fn grid_env_reset(env_ptr: *mut GridEnv) {
    let env = unsafe {
        assert!(!env_ptr.is_null());
        &mut *env_ptr
    };
    env.reset();
}

#[no_mangle]
pub extern "C" fn grid_env_get_state(env_ptr: *mut GridEnv) -> *mut Vec<(usize, usize, bool)> {
    let env = unsafe {
        assert!(!env_ptr.is_null());
        &mut *env_ptr
    };

    let grid_state = env.get_grid_state();
    Box::into_raw(Box::new(grid_state))
}

#[no_mangle]
pub extern "C" fn grid_env_free_state(state_ptr: *mut Vec<(usize, usize, bool)>) {
    if !state_ptr.is_null() {
        unsafe {
            let _ = Box::from_raw(state_ptr);
        }
    }
}