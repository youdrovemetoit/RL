import sys
import timeit
import ppo

def main(args):
    if len(args) < 2:
        print("Usage: PPO.py <environment name>")
        exit()

    env_name = args[1]
    print("Environment: ", env_name)

    ppoTrainerConfig = ppo.PPOTrainerConfig(
        num_envs = 1,
        seed = 0,
        device = ppo.DEFAULT_DEVICE,
        make_actor = lambda o, a: ppo.SimpleActor(o, a),
        make_critic = lambda o: ppo.SimpleCritic(o)
    )

    trainer = ppo.PPOTrainer(env_name, ppoTrainerConfig)

    trainConfig = ppo.PPOTrainerTrainConfig(
        total_steps = 50_000, #200_000
        rollout_steps = 512, # // ppoTrainerConfig.num_envs
        lr = 2.5e-4,
        adam_eps = 1e-5,
        normalise_observations = True,
        obs_clip = 10.0,
        checkpoint_every = 10
    )

    ppoConfig = ppo.PPOTrainerPPOUpdateConfig(
        clip_eps = 0.2,
        epochs = 4,
        minibatch_size = 64,
        norm_adv = True,
        clip_vloss = True,
        max_grad_norm = 0.5,
        ent_coef = 0.01,
        target_kl = 0.015,
        lam = 0.95    ,
        vf_coef = 0.5
    )

    start_time = timeit.default_timer()

    run = trainer.train(trainConfig, ppoConfig)['returns']
    elapsed = timeit.default_timer() - start_time
    print("Elapsed: ", elapsed)

if __name__ == "__main__":
    main(sys.argv)
