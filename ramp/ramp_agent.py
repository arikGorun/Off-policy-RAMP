class RAMPAgent:
    def __init__(self, env, rl_agent, aml, planner):
        self.env = env
        self.rl_agent = rl_agent
        self.aml = aml
        self.planner = planner
        self.trajectories = []

    def train(self, episodes=100):
        for ep in range(episodes):

            state, _ = self.env.reset()

            plan = self.planner.try_plan(
                state=state,
                goal=None,
                learned_model=self.aml.model
            )

            traj = []
            done = False

            while not done:

                if plan:
                    action = plan.pop(0)
                else:
                    action, _ = self.rl_agent.predict(
                        state,
                        deterministic=False
                    )

                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                traj.append(
                    (state, action, reward, next_state)
                )

                state = next_state

            self.trajectories.append(traj)

            self.aml.fit(self.trajectories)

            self.rl_agent.learn(
                total_timesteps=1000,
                reset_num_timesteps=False
            )
