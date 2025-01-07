use rand::prelude::SliceRandom;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

pub struct LineWorld {
    pos: usize,
}

impl LineWorld {
    pub fn new() -> Self {
        LineWorld {
            pos: 2
        }
    }

    pub fn num_states(&self) -> usize {
        5
    }

    pub fn num_actions(&self) -> usize {
        2
    }

    pub fn state_id(&self) -> usize {
        self.pos
    }

    pub fn reset(&mut self) {
        self.pos = 2;
    }

    pub fn is_game_over(&self) -> bool {
        self.pos == 0 || self.pos == 4
    }

    pub fn available_actions(&self) -> Vec<usize> {
        if self.is_game_over() {
            vec!()
        } else {
            vec!(0, 1)
        }
    }

    pub fn score(&self) -> f32 {
        match self.pos {
            0 => -1.0,
            4 => 1.0,
            _ => 0.0,
        }
    }

    pub fn step(&mut self, action: usize) {
        if self.is_game_over() {
            panic!("We are trying to play but game is over !")
        }

        if !self.available_actions().contains(&action) {
            panic!("Unauthorized action !")
        }

        match action {
            0 => {
                self.pos -= 1;
            }
            1 => {
                self.pos += 1;
            }
            _ => {
                unreachable!()
            }
        }
    }

    pub fn display(&self) {
        for s in 0usize..5 {
            if self.pos == s {
                print!("X");
            } else {
                print!("_");
            }
        }
        println!();
    }
}

pub fn q_learning(env: &mut LineWorld, max_episodes: usize, alpha: f32, epsilon: f32,
    gamma: f32,
) -> Vec<Vec<f32>> {
    let mut rng = Xoshiro256PlusPlus::from_entropy();

    let mut q = Vec::new();
    for _ in 0..env.num_states() {
        let mut v = Vec::new();
        for _ in 0..env.num_actions() {
            v.push(0.0f32);
        }
        q.push(v);
    }

    for _ in 0..max_episodes {
        env.reset();
        let mut s = env.state_id();
        let mut aa = env.available_actions();

        while !env.is_game_over() {

            let rnd_number = rand::random::<f32>();
            let a = if rnd_number <= epsilon {
                *aa.choose(&mut rng).unwrap()
            } else {
                let mut best_a = 0;
                let mut best_a_score = f32::MIN;

                for &a in aa.iter() {
                    let q_s_a = q[s][a];

                    if q_s_a >= best_a_score {
                        best_a = a;
                        best_a_score = q_s_a;
                    }
                }
                best_a
            };

            let prev_score = env.score();
            env.step(a);
            let r = env.score() - prev_score;

            let s_p = env.state_id();
            let aa_p = env.available_actions();

            let max_q_s_p = if env.is_game_over() {
                0.0f32
            } else {
                *q[s_p].iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap()
            };

            q[s][a] = q[s][a] + alpha * (r + gamma * max_q_s_p - q[s][a]);

            s = s_p;
            aa = aa_p;
        }
    }

    q.clone()
}


fn main() {
    let mut env = LineWorld::new();

    let q_star = q_learning(&mut env, 1000, 0.1, 1.0, 0.999);

    println!("{:?}", q_star);

}
