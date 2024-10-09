use std::ffi::c_void;

fn main() {
    unsafe {
        #[cfg(target_os = "linux")]
            let path = "./src/bin/envs/secret_env/libs/libsecret_envs.so";
        #[cfg(all(target_os = "macos", target_arch = "x86_64"))]
            let path = "./src/bin/envs/secret_env/libs/libsecret_envs_intel_macos.dylib";
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
            let path = "./src/bin/envs/secret_env/libs/libsecret_envs.dylib";
        #[cfg(windows)]
            let path = "./src/bin/envs/secret_env/libs/secret_envs.dll";
        let lib = libloading::Library::new(path).expect("Failed to load library");

        println!("Secret env 0 functions available for dynamic_programming ------------------------------------------------------");

        let secret_env_0_num_states: libloading::Symbol<unsafe extern fn() -> usize> =
            lib.get(b"secret_env_0_num_states").expect("Failed to load function `secret_env_0_num_states`");
        dbg!(secret_env_0_num_states());

        let secret_env_0_num_actions: libloading::Symbol<unsafe extern fn() -> usize> =
            lib.get(b"secret_env_0_num_actions").expect("Failed to load function `secret_env_0_num_actions`");
        dbg!(secret_env_0_num_actions());

        let secret_env_0_num_rewards: libloading::Symbol<unsafe extern fn() -> usize> =
            lib.get(b"secret_env_0_num_rewards").expect("Failed to load function `secret_env_0_num_rewards`");
        dbg!(secret_env_0_num_rewards());

        let secret_env_0_reward: libloading::Symbol<unsafe extern fn(usize) -> f32> =
            lib.get(b"secret_env_0_reward").expect("Failed to load function `secret_env_0_reward`");

        for i in 0..secret_env_0_num_rewards() {
            dbg!(secret_env_0_reward(i));
        }

        let secret_env_0_transition_probability: libloading::Symbol<unsafe extern fn(usize, usize, usize, usize) -> f32> =
            lib.get(b"secret_env_0_transition_probability").expect("Failed to load function `secret_env_0_transition_probability`");
        dbg!(secret_env_0_transition_probability(0, 0, 0, 0));

        println!("Secret env 0 functions available for Monte Carlo or TD or Methods ------------------------------------------------------");

        let secret_env_0_new: libloading::Symbol<unsafe extern fn() -> *mut c_void> =
            lib.get(b"secret_env_0_new").expect("Failed to load function `secret_env_0_new`");
        let env = secret_env_0_new();
        dbg!(env);

        let secret_env_0_reset: libloading::Symbol<unsafe extern fn(*mut c_void)> =
            lib.get(b"secret_env_0_reset").expect("Failed to load function `secret_env_0_reset`");
        secret_env_0_reset(env);

        let secret_env_0_state_id: libloading::Symbol<unsafe extern fn(*const c_void) -> usize> =
            lib.get(b"secret_env_0_state_id").expect("Failed to load function `secret_env_0_state_id`");
        dbg!(secret_env_0_state_id(env));

        let secret_env_0_is_forbidden: libloading::Symbol<unsafe extern fn(*const c_void, usize) -> bool> =
            lib.get(b"secret_env_0_is_forbidden").expect("Failed to load function `secret_env_0_is_forbidden`");
        dbg!(secret_env_0_is_forbidden(env, 0));

        let secret_env_0_is_game_over: libloading::Symbol<unsafe extern fn(*const c_void) -> bool> =
            lib.get(b"secret_env_0_is_game_over").expect("Failed to load function `secret_env_0_is_game_over`");
        dbg!(secret_env_0_is_game_over(env));

        while !secret_env_0_is_game_over(env) {
            let secret_env_0_display: libloading::Symbol<unsafe extern fn(*const c_void)> =
                lib.get(b"secret_env_0_display").expect("Failed to load function `secret_env_0_display`");
            dbg!(secret_env_0_display(env));
            
            let secret_env_0_available_actions: libloading::Symbol<unsafe extern fn(*const c_void) -> *const usize> =
                lib.get(b"secret_env_0_available_actions").expect("Failed to load function `secret_env_0_available_actions`");
            let actions = secret_env_0_available_actions(env);
            dbg!(actions);

            let secret_env_0_available_actions_len: libloading::Symbol<unsafe extern fn(*const c_void) -> usize> =
                lib.get(b"secret_env_0_available_actions_len").expect("Failed to load function `secret_env_0_available_actions_len`");
            dbg!(secret_env_0_available_actions_len(env));

            // show all available actions
            for i in 0..secret_env_0_available_actions_len(env) {
                dbg!(*actions.add(i));
            }

            let first_available_action = *actions;

            let secret_env_0_available_actions_delete: libloading::Symbol<unsafe extern fn(*const usize, usize)> =
                lib.get(b"secret_env_0_available_actions_delete").expect("Failed to load function `secret_env_0_available_actions_delete`");
            secret_env_0_available_actions_delete(actions, secret_env_0_available_actions_len(env));

            let secret_env_0_step: libloading::Symbol<unsafe extern fn(*mut c_void, usize)> =
                lib.get(b"secret_env_0_step").expect("Failed to load function `secret_env_0_step`");
            secret_env_0_step(env, first_available_action);

            let secret_env_0_state_id: libloading::Symbol<unsafe extern fn(*const c_void) -> usize> =
                lib.get(b"secret_env_0_state_id").expect("Failed to load function `secret_env_0_state_id`");
            dbg!(secret_env_0_state_id(env));
        }

        let secret_env_0_display: libloading::Symbol<unsafe extern fn(*const c_void)> =
            lib.get(b"secret_env_0_display").expect("Failed to load function `secret_env_0_display`");
        dbg!(secret_env_0_display(env));

        let secret_env_0_score: libloading::Symbol<unsafe extern fn(*const c_void) -> f32> =
            lib.get(b"secret_env_0_score").expect("Failed to load function `secret_env_0_score`");
        dbg!(secret_env_0_score(env));

        let secret_env_0_delete: libloading::Symbol<unsafe extern fn(*mut c_void)> =
            lib.get(b"secret_env_0_delete").expect("Failed to load function `secret_env_0_delete`");
        secret_env_0_delete(env);


        let secret_env_0_from_random_state: libloading::Symbol<unsafe extern fn() -> *mut c_void> =
            lib.get(b"secret_env_0_from_random_state").expect("Failed to load function `secret_env_0_from_random_state`");
        let env = secret_env_0_from_random_state();
        dbg!(env);

        secret_env_0_delete(env);

        println!("Secret env 1 functions available for dynamic_programming ------------------------------------------------------");

        let secret_env_1_num_states: libloading::Symbol<unsafe extern fn() -> usize> =
            lib.get(b"secret_env_1_num_states").expect("Failed to load function `secret_env_1_num_states`");
        dbg!(secret_env_1_num_states());

        let secret_env_1_num_actions: libloading::Symbol<unsafe extern fn() -> usize> =
            lib.get(b"secret_env_1_num_actions").expect("Failed to load function `secret_env_1_num_actions`");
        dbg!(secret_env_1_num_actions());

        let secret_env_1_num_rewards: libloading::Symbol<unsafe extern fn() -> usize> =
            lib.get(b"secret_env_1_num_rewards").expect("Failed to load function `secret_env_1_num_rewards`");
        dbg!(secret_env_1_num_rewards());

        let secret_env_1_reward: libloading::Symbol<unsafe extern fn(usize) -> f32> =
            lib.get(b"secret_env_1_reward").expect("Failed to load function `secret_env_1_reward`");

        for i in 0..secret_env_1_num_rewards() {
            dbg!(secret_env_1_reward(i));
        }

        let secret_env_1_transition_probability: libloading::Symbol<unsafe extern fn(usize, usize, usize, usize) -> f32> =
            lib.get(b"secret_env_1_transition_probability").expect("Failed to load function `secret_env_1_transition_probability`");
        dbg!(secret_env_1_transition_probability(0, 0, 0, 0));

        println!("Secret env 1 functions available for Monte Carlo or TD Methods ------------------------------------------------------");

        let secret_env_1_new: libloading::Symbol<unsafe extern fn() -> *mut c_void> =
            lib.get(b"secret_env_1_new").expect("Failed to load function `secret_env_1_new`");
        let env = secret_env_1_new();
        dbg!(env);

        let secret_env_1_reset: libloading::Symbol<unsafe extern fn(*mut c_void)> =
            lib.get(b"secret_env_1_reset").expect("Failed to load function `secret_env_1_reset`");
        secret_env_1_reset(env);

        let secret_env_1_state_id: libloading::Symbol<unsafe extern fn(*const c_void) -> usize> =
            lib.get(b"secret_env_1_state_id").expect("Failed to load function `secret_env_1_state_id`");
        dbg!(secret_env_1_state_id(env));

        let secret_env_1_is_forbidden: libloading::Symbol<unsafe extern fn(*const c_void, usize) -> bool> =
            lib.get(b"secret_env_1_is_forbidden").expect("Failed to load function `secret_env_1_is_forbidden`");
        dbg!(secret_env_1_is_forbidden(env, 0));

        let secret_env_1_is_game_over: libloading::Symbol<unsafe extern fn(*const c_void) -> bool> =
            lib.get(b"secret_env_1_is_game_over").expect("Failed to load function `secret_env_1_is_game_over`");
        dbg!(secret_env_1_is_game_over(env));

        while !secret_env_1_is_game_over(env) {
            let secret_env_1_display: libloading::Symbol<unsafe extern fn(*const c_void)> =
                lib.get(b"secret_env_1_display").expect("Failed to load function `secret_env_1_display`");
            dbg!(secret_env_1_display(env));
            
            let secret_env_1_available_actions: libloading::Symbol<unsafe extern fn(*const c_void) -> *const usize> =
                lib.get(b"secret_env_1_available_actions").expect("Failed to load function `secret_env_1_available_actions`");
            let actions = secret_env_1_available_actions(env);
            dbg!(actions);

            let secret_env_1_available_actions_len: libloading::Symbol<unsafe extern fn(*const c_void) -> usize> =
                lib.get(b"secret_env_1_available_actions_len").expect("Failed to load function `secret_env_1_available_actions_len`");
            dbg!(secret_env_1_available_actions_len(env));

            // show all available actions
            for i in 0..secret_env_1_available_actions_len(env) {
                dbg!(*actions.add(i));
            }

            let first_available_action = *actions;

            let secret_env_1_available_actions_delete: libloading::Symbol<unsafe extern fn(*const usize, usize)> =
                lib.get(b"secret_env_1_available_actions_delete").expect("Failed to load function `secret_env_1_available_actions_delete`");
            secret_env_1_available_actions_delete(actions, secret_env_1_available_actions_len(env));

            let secret_env_1_step: libloading::Symbol<unsafe extern fn(*mut c_void, usize)> =
                lib.get(b"secret_env_1_step").expect("Failed to load function `secret_env_1_step`");
            secret_env_1_step(env, first_available_action);

            let secret_env_1_state_id: libloading::Symbol<unsafe extern fn(*const c_void) -> usize> =
                lib.get(b"secret_env_1_state_id").expect("Failed to load function `secret_env_1_state_id`");
            dbg!(secret_env_1_state_id(env));
        }
        let secret_env_1_display: libloading::Symbol<unsafe extern fn(*const c_void)> =
            lib.get(b"secret_env_1_display").expect("Failed to load function `secret_env_1_display`");
        dbg!(secret_env_1_display(env));

        let secret_env_1_score: libloading::Symbol<unsafe extern fn(*const c_void) -> f32> =
            lib.get(b"secret_env_1_score").expect("Failed to load function `secret_env_1_score`");
        dbg!(secret_env_1_score(env));

        let secret_env_1_delete: libloading::Symbol<unsafe extern fn(*mut c_void)> =
            lib.get(b"secret_env_1_delete").expect("Failed to load function `secret_env_1_delete`");
        secret_env_1_delete(env);

        let secret_env_1_from_random_state: libloading::Symbol<unsafe extern fn() -> *mut c_void> =
            lib.get(b"secret_env_1_from_random_state").expect("Failed to load function `secret_env_1_from_random_state`");
        let env = secret_env_1_from_random_state();
        dbg!(env);

        secret_env_1_delete(env);

        println!("Secret env 2 functions available for dynamic_programming ------------------------------------------------------");

        let secret_env_2_num_states: libloading::Symbol<unsafe extern fn() -> usize> =
            lib.get(b"secret_env_2_num_states").expect("Failed to load function `secret_env_2_num_states`");
        dbg!(secret_env_2_num_states());

        let secret_env_2_num_actions: libloading::Symbol<unsafe extern fn() -> usize> =
            lib.get(b"secret_env_2_num_actions").expect("Failed to load function `secret_env_2_num_actions`");
        dbg!(secret_env_2_num_actions());

        let secret_env_2_num_rewards: libloading::Symbol<unsafe extern fn() -> usize> =
            lib.get(b"secret_env_2_num_rewards").expect("Failed to load function `secret_env_2_num_rewards`");
        dbg!(secret_env_2_num_rewards());

        let secret_env_2_reward: libloading::Symbol<unsafe extern fn(usize) -> f32> =
            lib.get(b"secret_env_2_reward").expect("Failed to load function `secret_env_2_reward`");

        for i in 0..secret_env_2_num_rewards() {
            dbg!(secret_env_2_reward(i));
        }

        let secret_env_2_transition_probability: libloading::Symbol<unsafe extern fn(usize, usize, usize, usize) -> f32> =
            lib.get(b"secret_env_2_transition_probability").expect("Failed to load function `secret_env_2_transition_probability`");
        dbg!(secret_env_2_transition_probability(0, 0, 0, 0));

        println!("Secret env 2 functions available for Monte Carlo or TD Methods ------------------------------------------------------");

        let secret_env_2_new: libloading::Symbol<unsafe extern fn() -> *mut c_void> =
            lib.get(b"secret_env_2_new").expect("Failed to load function `secret_env_2_new`");
        let env = secret_env_2_new();
        dbg!(env);

        let secret_env_2_reset: libloading::Symbol<unsafe extern fn(*mut c_void)> =
            lib.get(b"secret_env_2_reset").expect("Failed to load function `secret_env_2_reset`");
        secret_env_2_reset(env);

        let secret_env_2_state_id: libloading::Symbol<unsafe extern fn(*const c_void) -> usize> =
            lib.get(b"secret_env_2_state_id").expect("Failed to load function `secret_env_2_state_id`");
        dbg!(secret_env_2_state_id(env));

        let secret_env_2_is_forbidden: libloading::Symbol<unsafe extern fn(*const c_void, usize) -> bool> =
            lib.get(b"secret_env_2_is_forbidden").expect("Failed to load function `secret_env_2_is_forbidden`");
        dbg!(secret_env_2_is_forbidden(env, 0));

        let secret_env_2_is_game_over: libloading::Symbol<unsafe extern fn(*const c_void) -> bool> =
            lib.get(b"secret_env_2_is_game_over").expect("Failed to load function `secret_env_2_is_game_over`");
        dbg!(secret_env_2_is_game_over(env));

        while !secret_env_2_is_game_over(env) {
            let secret_env_2_display: libloading::Symbol<unsafe extern fn(*const c_void)> =
                lib.get(b"secret_env_2_display").expect("Failed to load function `secret_env_2_display`");
            dbg!(secret_env_2_display(env));

            let secret_env_2_available_actions: libloading::Symbol<unsafe extern fn(*const c_void) -> *const usize> =
                lib.get(b"secret_env_2_available_actions").expect("Failed to load function `secret_env_2_available_actions`");
            let actions = secret_env_2_available_actions(env);
            dbg!(actions);

            let secret_env_2_available_actions_len: libloading::Symbol<unsafe extern fn(*const c_void) -> usize> =
                lib.get(b"secret_env_2_available_actions_len").expect("Failed to load function `secret_env_2_available_actions_len`");
            dbg!(secret_env_2_available_actions_len(env));

            // show all available actions
            for i in 0..secret_env_2_available_actions_len(env) {
                dbg!(*actions.add(i));
            }

            let first_available_action = *actions;

            let secret_env_2_available_actions_delete: libloading::Symbol<unsafe extern fn(*const usize, usize)> =
                lib.get(b"secret_env_2_available_actions_delete").expect("Failed to load function `secret_env_2_available_actions_delete`");
            secret_env_2_available_actions_delete(actions, secret_env_2_available_actions_len(env));

            let secret_env_2_step: libloading::Symbol<unsafe extern fn(*mut c_void, usize)> =
                lib.get(b"secret_env_2_step").expect("Failed to load function `secret_env_2_step`");
            secret_env_2_step(env, first_available_action);

            let secret_env_2_state_id: libloading::Symbol<unsafe extern fn(*const c_void) -> usize> =
                lib.get(b"secret_env_2_state_id").expect("Failed to load function `secret_env_2_state_id`");
            dbg!(secret_env_2_state_id(env));
        }
        let secret_env_2_display: libloading::Symbol<unsafe extern fn(*const c_void)> =
            lib.get(b"secret_env_2_display").expect("Failed to load function `secret_env_2_display`");
        dbg!(secret_env_2_display(env));

        let secret_env_2_score: libloading::Symbol<unsafe extern fn(*const c_void) -> f32> =
            lib.get(b"secret_env_2_score").expect("Failed to load function `secret_env_2_score`");
        dbg!(secret_env_2_score(env));

        let secret_env_2_delete: libloading::Symbol<unsafe extern fn(*mut c_void)> =
            lib.get(b"secret_env_2_delete").expect("Failed to load function `secret_env_2_delete`");
        secret_env_2_delete(env);

        let secret_env_2_from_random_state: libloading::Symbol<unsafe extern fn() -> *mut c_void> =
            lib.get(b"secret_env_2_from_random_state").expect("Failed to load function `secret_env_2_from_random_state`");
        let env = secret_env_2_from_random_state();
        dbg!(env);

        secret_env_2_delete(env);

        println!("Secret env 3 functions available for dynamic_programming ------------------------------------------------------");

        let secret_env_3_num_states: libloading::Symbol<unsafe extern fn() -> usize> =
            lib.get(b"secret_env_3_num_states").expect("Failed to load function `secret_env_3_num_states`");
        dbg!(secret_env_3_num_states());

        let secret_env_3_num_actions: libloading::Symbol<unsafe extern fn() -> usize> =
            lib.get(b"secret_env_3_num_actions").expect("Failed to load function `secret_env_3_num_actions`");
        dbg!(secret_env_3_num_actions());

        let secret_env_3_num_rewards: libloading::Symbol<unsafe extern fn() -> usize> =
            lib.get(b"secret_env_3_num_rewards").expect("Failed to load function `secret_env_3_num_rewards`");
        dbg!(secret_env_3_num_rewards());

        let secret_env_3_reward: libloading::Symbol<unsafe extern fn(usize) -> f32> =
            lib.get(b"secret_env_3_reward").expect("Failed to load function `secret_env_3_reward`");

        for i in 0..secret_env_3_num_rewards() {
            dbg!(secret_env_3_reward(i));
        }

        let secret_env_3_transition_probability: libloading::Symbol<unsafe extern fn(usize, usize, usize, usize) -> f32> =
            lib.get(b"secret_env_3_transition_probability").expect("Failed to load function `secret_env_3_transition_probability`");
        dbg!(secret_env_3_transition_probability(0, 0, 0, 0));

        println!("Secret env 3 functions available for Monte Carlo or TD Methods ------------------------------------------------------");

        let secret_env_3_new: libloading::Symbol<unsafe extern fn() -> *mut c_void> =
            lib.get(b"secret_env_3_new").expect("Failed to load function `secret_env_3_new`");
        let env = secret_env_3_new();
        dbg!(env);

        let secret_env_3_reset: libloading::Symbol<unsafe extern fn(*mut c_void)> =
            lib.get(b"secret_env_3_reset").expect("Failed to load function `secret_env_3_reset`");
        secret_env_3_reset(env);

        let secret_env_3_state_id: libloading::Symbol<unsafe extern fn(*const c_void) -> usize> =
            lib.get(b"secret_env_3_state_id").expect("Failed to load function `secret_env_3_state_id`");
        dbg!(secret_env_3_state_id(env));

        let secret_env_3_is_forbidden: libloading::Symbol<unsafe extern fn(*const c_void, usize) -> bool> =
            lib.get(b"secret_env_3_is_forbidden").expect("Failed to load function `secret_env_3_is_forbidden`");
        dbg!(secret_env_3_is_forbidden(env, 0));

        let secret_env_3_is_game_over: libloading::Symbol<unsafe extern fn(*const c_void) -> bool> =
            lib.get(b"secret_env_3_is_game_over").expect("Failed to load function `secret_env_3_is_game_over`");
        dbg!(secret_env_3_is_game_over(env));

        while !secret_env_3_is_game_over(env) {
            let secret_env_3_display: libloading::Symbol<unsafe extern fn(*const c_void)> =
                lib.get(b"secret_env_3_display").expect("Failed to load function `secret_env_3_display`");
            dbg!(secret_env_3_display(env));

            let secret_env_3_available_actions: libloading::Symbol<unsafe extern fn(*const c_void) -> *const usize> =
                lib.get(b"secret_env_3_available_actions").expect("Failed to load function `secret_env_3_available_actions`");
            let actions = secret_env_3_available_actions(env);
            dbg!(actions);

            let secret_env_3_available_actions_len: libloading::Symbol<unsafe extern fn(*const c_void) -> usize> =
                lib.get(b"secret_env_3_available_actions_len").expect("Failed to load function `secret_env_3_available_actions_len`");
            dbg!(secret_env_3_available_actions_len(env));

            // show all available actions
            for i in 0..secret_env_3_available_actions_len(env) {
                dbg!(*actions.add(i));
            }

            let first_available_action = *actions;

            let secret_env_3_available_actions_delete: libloading::Symbol<unsafe extern fn(*const usize, usize)> =
                lib.get(b"secret_env_3_available_actions_delete").expect("Failed to load function `secret_env_3_available_actions_delete`");
            secret_env_3_available_actions_delete(actions, secret_env_3_available_actions_len(env));

            let secret_env_3_step: libloading::Symbol<unsafe extern fn(*mut c_void, usize)> =
                lib.get(b"secret_env_3_step").expect("Failed to load function `secret_env_3_step`");
            secret_env_3_step(env, first_available_action);

            let secret_env_3_state_id: libloading::Symbol<unsafe extern fn(*const c_void) -> usize> =
                lib.get(b"secret_env_3_state_id").expect("Failed to load function `secret_env_3_state_id`");
            dbg!(secret_env_3_state_id(env));
        }
        let secret_env_3_display: libloading::Symbol<unsafe extern fn(*const c_void)> =
            lib.get(b"secret_env_3_display").expect("Failed to load function `secret_env_3_display`");
        dbg!(secret_env_3_display(env));

        let secret_env_3_score: libloading::Symbol<unsafe extern fn(*const c_void) -> f32> =
            lib.get(b"secret_env_3_score").expect("Failed to load function `secret_env_3_score`");
        dbg!(secret_env_3_score(env));

        let secret_env_3_delete: libloading::Symbol<unsafe extern fn(*mut c_void)> =
            lib.get(b"secret_env_3_delete").expect("Failed to load function `secret_env_3_delete`");
        secret_env_3_delete(env);

        let secret_env_3_from_random_state: libloading::Symbol<unsafe extern fn() -> *mut c_void> =
            lib.get(b"secret_env_3_from_random_state").expect("Failed to load function `secret_env_3_from_random_state`");
        let env = secret_env_3_from_random_state();
        dbg!(env);

        secret_env_3_delete(env);
    };
}
