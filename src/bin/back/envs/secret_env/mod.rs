use libloading::Library;
use nalgebra::DVector;
use std::ffi::c_void;
use std::sync::Arc;
use crate::back::envs::basic_env::Env;

pub struct SecretEnv {
    lib: Arc<Library>,
    env: *mut c_void,
    env_name: String,
}

impl SecretEnv {
    /// Constructor to initialize SecretEnv with a dynamic `env_name`
    pub unsafe fn new(env_name: &str) -> Self {
        // Determine the library path based on the OS
        #[cfg(target_os = "linux")]
        let path = "./libs/libsecret_envs.so";
        #[cfg(all(target_os = "macos", target_arch = "x86_64"))]
        let path = "./libs/libsecret_envs_intel_macos.dylib";
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        let path = "./libs/libsecret_envs.dylib";
        #[cfg(windows)]
        let path = "./libs/secret_envs.dll";

        // Convert the env_name to a String
        let env_name = env_name.to_string();

        // Load the library
        let lib = Arc::new(Library::new(path).expect("Failed to load library"));

        // Load the dynamic function based on `env_name`
        let secret_env_new: libloading::Symbol<unsafe extern fn() -> *mut c_void> = unsafe {
            lib.get(env_name.as_bytes())
                .expect(&format!("Failed to load `{}`", env_name))
        };

        // Call the function to create the environment
        let env = unsafe { secret_env_new() };

        SecretEnv { lib, env, env_name }
    }

    /// Delete the dynamically loaded environment
    pub fn delete(&mut self) {
        let delete_function_name = format!("{}_delete", self.env_name.trim_end_matches("_new"));
        let secret_env_delete: libloading::Symbol<unsafe extern fn(*mut c_void)> = unsafe {
            self.lib
                .get(delete_function_name.as_bytes())
                .expect(&format!("Failed to load `{}`", delete_function_name))
        };
        unsafe {
            secret_env_delete(self.env);
        }
    }
}

impl Drop for SecretEnv {
    fn drop(&mut self) {
        self.delete();
    }
}

// Implement the `Env` trait for `SecretEnv`
impl Env for SecretEnv {
    // Implement other required methods...

    fn num_states(&self) -> usize {
        let num_states_function_name = format!("{}_num_states", self.env_name.trim_end_matches("_new"));
        let secret_env_num_states: libloading::Symbol<unsafe extern fn() -> usize> = unsafe {
            self.lib
                .get(num_states_function_name.as_bytes())
                .expect(&format!("Failed to load `{}`", num_states_function_name))
        };
        unsafe { secret_env_num_states() }
    }

    fn num_actions(&self) -> usize {
        let num_actions_function_name = format!("{}_num_actions", self.env_name.trim_end_matches("_new"));
        let secret_env_num_actions: libloading::Symbol<unsafe extern fn() -> usize> = unsafe {
            self.lib
                .get(num_actions_function_name.as_bytes())
                .expect(&format!("Failed to load `{}`", num_actions_function_name))
        };
        unsafe { secret_env_num_actions() }
    }

    fn num_rewards(&self) -> usize {
        let num_rewards_function_name = format!("{}_num_rewards", self.env_name.trim_end_matches("_new"));
        let secret_env_0_num_rewards: libloading::Symbol<unsafe extern fn() -> usize> = unsafe {
            self.lib
                .get(num_rewards_function_name.as_bytes())
                .expect("Failed to load `secret_env_0_num_rewards`")
        };
        unsafe { secret_env_0_num_rewards() }
    }

    fn get_reward_vector(&self) -> Vec<f32> {
        panic!("Not yet implemented");
    }

    fn get_terminal_states(&self) -> Vec<usize> {
        panic!("Not yet implemented");
    }

    fn get_reward(&self, num: usize) -> f32 {
        let reward_function_name = format!("{}_reward", self.env_name.trim_end_matches("_new"));
        let secret_env_reward: libloading::Symbol<unsafe extern fn(usize) -> f32> = unsafe {
            self.lib
                .get(reward_function_name.as_bytes())
                .expect(&format!("Failed to load `{}`", reward_function_name))
        };
        unsafe { secret_env_reward(num) }
    }

    fn get_action_spaces(&self) -> Vec<usize> {
        panic!("Not yet implemented");
    }

    fn p(&self, s: i32, a: i32, s_p: i32, r_index: i32) -> f32 {
        let transition_probability_function_name =
            format!("{}_transition_probability", self.env_name.trim_end_matches("_new"));
        let secret_env_transition_probability: libloading::Symbol<
            unsafe extern fn(usize, usize, usize, usize) -> f32,
        > = unsafe {
            self.lib
                .get(transition_probability_function_name.as_bytes())
                .expect(&format!(
                    "Failed to load `{}`",
                    transition_probability_function_name
                ))
        };
        unsafe { secret_env_transition_probability(s as usize, a as usize, s_p as usize, r_index as usize) }
    }

    fn state_id(&self) -> usize {
        let state_id_function_name = format!("{}_state_id", self.env_name.trim_end_matches("_new"));
        let secret_env_state_id: libloading::Symbol<unsafe extern fn(*const c_void) -> usize> = unsafe {
            self.lib
                .get(state_id_function_name.as_bytes())
                .expect(&format!("Failed to load `{}`", state_id_function_name))
        };
        unsafe { secret_env_state_id(self.env) }
    }

    fn reset(&mut self) {
        let reset_function_name = format!("{}_reset", self.env_name.trim_end_matches("_new"));
        let secret_env_reset: libloading::Symbol<unsafe extern fn(*mut c_void)> = unsafe {
            self.lib
                .get(reset_function_name.as_bytes())
                .expect(&format!("Failed to load `{}`", reset_function_name))
        };
        unsafe {
            secret_env_reset(self.env);
        }
    }

    fn display(&self) {
        let display_function_name = format!("{}_display", self.env_name.trim_end_matches("_new"));
        let secret_env_display: libloading::Symbol<unsafe extern fn(*const c_void)> = unsafe {
            self.lib
                .get(display_function_name.as_bytes())
                .expect(&format!("Failed to load `{}`", display_function_name))
        };
        unsafe {
            secret_env_display(self.env);
        }
    }

    fn is_forbidden(&self, action: usize) -> bool {
        let is_forbidden_function_name =
            format!("{}_is_forbidden", self.env_name.trim_end_matches("_new"));
        let secret_env_is_forbidden: libloading::Symbol<unsafe extern fn(*const c_void, usize) -> bool> =
            unsafe {
                self.lib
                    .get(is_forbidden_function_name.as_bytes())
                    .expect(&format!("Failed to load `{}`", is_forbidden_function_name))
            };
        unsafe { secret_env_is_forbidden(self.env, action) }
    }

    fn is_game_over(&self) -> bool {
        let is_game_over_function_name =
            format!("{}_is_game_over", self.env_name.trim_end_matches("_new"));
        let secret_env_is_game_over: libloading::Symbol<unsafe extern fn(*const c_void) -> bool> = unsafe {
            self.lib
                .get(is_game_over_function_name.as_bytes())
                .expect(&format!("Failed to load `{}`", is_game_over_function_name))
        };
        unsafe { secret_env_is_game_over(self.env) }
    }

    fn available_actions(&self) -> DVector<i32> {
        let available_actions_function_name =
            format!("{}_available_actions", self.env_name.trim_end_matches("_new"));
        let available_actions_len_function_name =
            format!("{}_available_actions_len", self.env_name.trim_end_matches("_new"));
        let available_actions_delete_function_name =
            format!("{}_available_actions_delete", self.env_name.trim_end_matches("_new"));

        let secret_env_available_actions: libloading::Symbol<unsafe extern fn(*const c_void) -> *const usize> =
            unsafe {
                self.lib
                    .get(available_actions_function_name.as_bytes())
                    .expect(&format!("Failed to load `{}`", available_actions_function_name))
            };

        let secret_env_available_actions_len: libloading::Symbol<unsafe extern fn(*const c_void) -> usize> =
            unsafe {
                self.lib
                    .get(available_actions_len_function_name.as_bytes())
                    .expect(&format!("Failed to load `{}`", available_actions_len_function_name))
            };

        let secret_env_available_actions_delete: libloading::Symbol<unsafe extern fn(*const usize, usize)> =
            unsafe {
                self.lib
                    .get(available_actions_delete_function_name.as_bytes())
                    .expect(&format!("Failed to load `{}`", available_actions_delete_function_name))
            };

        let actions_ptr = unsafe { secret_env_available_actions(self.env) };
        let actions_len = unsafe { secret_env_available_actions_len(self.env) };

        let actions: Vec<i32> = (0..actions_len)
            .map(|i| unsafe { *(actions_ptr.add(i)) } as i32)
            .collect();

        // Clean up memory for available actions
        unsafe {
            secret_env_available_actions_delete(actions_ptr, actions_len);
        }

        DVector::from_vec(actions)
    }

    fn step(&mut self, action: i32) {
        let step_function_name = format!("{}_step", self.env_name.trim_end_matches("_new"));
        let secret_env_step: libloading::Symbol<unsafe extern fn(*mut c_void, usize)> = unsafe {
            self.lib
                .get(step_function_name.as_bytes())
                .expect(&format!("Failed to load `{}`", step_function_name))
        };
        unsafe {
            secret_env_step(self.env, action as usize);
        }
    }

    fn score(&self) -> f32 {
        let score_function_name = format!("{}_score", self.env_name.trim_end_matches("_new"));
        let secret_env_score: libloading::Symbol<unsafe extern fn(*const c_void) -> f32> = unsafe {
            self.lib
                .get(score_function_name.as_bytes())
                .expect(&format!("Failed to load `{}`", score_function_name))
        };
        unsafe { secret_env_score(self.env) }
    }

    fn from_random_state() -> Self
    where
        Self: Sized
    {
        panic!("Not yet implemented");
    }

    fn transition_probability(&self, s: usize, a: usize, s_p: usize, r_index: usize) -> f32 {
        // Construct the transition probability function name dynamically
        let prob_function_name = format!("{}_transition_probability", self.env_name.trim_end_matches("_new"));
        let secret_env_transition_probability: libloading::Symbol<
            unsafe extern fn(usize, usize, usize, usize) -> f32,
        > = unsafe {
            self.lib
                .get(prob_function_name.as_bytes())
                .expect(&format!("Failed to load `{}`", prob_function_name))
        };

        // Call the function and return the result
        unsafe { secret_env_transition_probability(s, a, s_p, r_index) }
    }
}
