mod envs;
mod rl;

use envs::line_world_env::LineEnv;
use envs::line_world_env::Env;

fn main() {
    // Create a 3x3 matrix and a 3D vector

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

    rl::some_rl_function()
}