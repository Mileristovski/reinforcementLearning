use crate::back::rl::dynamic_programming::iterative_policy_eval::iterative_policy_evaluation;
use crate::back::rl::dynamic_programming::policy_iteration::policy_iteration;
use crate::back::rl::dynamic_programming::value_iteration::value_iteration;

pub fn testing_iterative_policy_evaluation() {
    let s = vec![0, 1, 2, 3, 4];
    let a = vec![0, 1];
    let r = vec![-1.0, 0.0, 1.0];
    let t = vec![0, 4];

    let mut pi_right = vec![vec![0.0; a.len()]; s.len()];
    // Set the second column (index 1) to 1.0
    for state in 0..s.len() {
        pi_right[state][1] = 1.0;
    }

    let mut p = vec![vec![vec![vec![0.0f32; r.len()]; s.len()]; a.len()]; s.len()];

    // Setting up the probability transition matrix p
    for &s_p in &s {
        if s_p == 0 || s_p == 4 {
            continue;
        }
        for &action in &a {
            if action == 0 && s_p > 1 {
                p[s_p][action][s_p - 1][1] = 1.0;
            }
            if action == 1 && s_p < 3 {
                p[s_p][action][s_p + 1][1] = 1.0;
            }
        }
    }

    // Set specific probabilities
    p[1][0][0][0] = 1.0;
    p[3][1][4][2] = 1.0;
    println!("{:?}", p);
    let gamma = 0.999;
    let theta = 0.0001;

    let v = iterative_policy_evaluation(&pi_right, &s, &a, &r, &p, &t, gamma, theta);
    println!("{:?}", v);
}

pub fn testing_policy_iterations() {
    let s = vec![0usize, 1, 2, 3, 4];
    let a = vec![0usize, 1];
    let r = vec![-1.0f32, 0.0, 1.0];
    let t = vec![0usize, 4];

    let mut p = vec![vec![vec![vec![0.0f32; r.len()]; s.len()]; a.len()]; s.len()];

    for &s_p in &s {
        if s_p == 0 || s_p == 4 {
            continue;
        }
        for &action in &a {
            if action == 0 && s_p > 1 {
                p[s_p][action][s_p - 1][1] = 1.0;
            }
            if action == 1 && s_p < 3 {
                p[s_p][action][s_p + 1][1] = 1.0;
            }
        }
    }

    // Set probabilities
    p[1][0][0][0] = 1.0;
    p[3][1][4][2] = 1.0;

    let gamma = 0.999;
    let theta = 0.0001;

    // Call policy iteration
    let v = policy_iteration(&s, &a, &r, &p, &t, gamma, theta);
    println!("{:?}", v);
}

pub fn testing_value_iteration() {
    let s = vec![0usize, 1, 2, 3, 4];
    let a = vec![0usize, 1];
    let r = vec![-1.0f32, 0.0, 1.0];
    let t = vec![0usize, 4];

    let mut p = vec![vec![vec![vec![0.0f32; r.len()]; s.len()]; a.len()]; s.len()];

    for &s_p in &s {
        if s_p == 0 || s_p == 4 {
            continue;
        }
        for &action in &a {
            if action == 0 && s_p > 1 {
                p[s_p][action][s_p - 1][1] = 1.0;
            }
            if action == 1 && s_p < 3 {
                p[s_p][action][s_p + 1][1] = 1.0;
            }
        }
    }

    // Set probabilities
    p[1][0][0][0] = 1.0;
    p[3][1][4][2] = 1.0;

    let gamma = 0.999;
    let theta = 0.0001;

    // Call value iteration
    let (v, pi) = value_iteration(&s, &a, &r, &p, &t, gamma, theta);
    println!("Optimal Values: {:?}", v);
    println!("Optimal Policy: {:?}", pi);
}