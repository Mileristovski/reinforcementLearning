mod envs;
mod rl;

use envs::line_world_env::LineEnv;
use envs::line_world_env::Env;
use rl::dynamic_programming::policy_iteration::policy_iteration;

fn test_env() {
    let s = vec![0, 1, 2, 3, 4];
    let a = vec![0, 1];
    let r = vec![-1, 0, 1];
    let t = vec![0, 4];


    // Create an instance of LinearAlgebra
    let mut env = LineEnv::new(s, a, r, t);

    // Display matrix and vector
    env.display();

    env.step(1);
    env.display();
    env.score();

    env.step(1);
    env.display();
    env.score();
}

fn main() {
    // Test the line world env
    // test_env();

    // Example usage
    let s = vec![0, 1, 2, 3, 4];
    let a = vec![0, 1];
    let r = vec![-1.0, 0.0, 1.0];
    let t = vec![0, 4];

    let mut p = vec![vec![vec![vec![0.0; r.len()]; s.len()]; a.len()]; s.len()];

    for &s_p in &s {
        if s_p == 0 || s_p == 4 {
            continue;
        }
        for &a in &a {
            if a == 0 && s_p > 1 {
                p[s_p][a][s_p - 1][1] = 1.0;
            }
            if a == 1 && s_p < 3 {
                p[s_p][a][s_p + 1][1] = 1.0;
            }
        }
    }

    // Set specific probabilities
    p[1][0][0][0] = 1.0;
    p[3][1][4][2] = 1.0;


    let mut pi_right = vec![vec![0.0; a.len()]; s.len()];
    // Set the second column (index 1) to 1.0
    for state in 0..s.len() {
        pi_right[state][1] = 1.0;
    }

    let gamma = 0.999;
    let theta = 0.0001;

    let v = policy_iteration(&s, &a, &r, &p, &t, gamma, theta);
    println!("{:?}", v);
}