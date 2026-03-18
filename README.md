# Reinforcement Learning VS Optimization

The Constraints are defined as follows:

- Optimization Problem: Maximize the reward function while satisfying the constraints.
- Safe Constraints: Ensure that the agent's actions do not violate certain safety constraints, even if it means sacrificing some reward.

## Single Constraint Version

Target: Maximize the Average Reward (AR) while satisfying the safety constraints.

  <table>
      <thead>
          <tr>
              <th>Folder</th>
              <th>Description</th>
          </tr>
      </thead>
      <tbody>
          <tr>
              <td><a href="./problem1">Problem 1</a></td>
              <td>Reinforcement Learning Environment</td>
          </tr>
          <tr>
              <td><a href="./problem2_single">Problem 2</a></td>
              <td>ILP Problem</td>
          </tr>
          <tr>
              <td><a href="./problem3_single">Problem 3</a></td>
              <td>Reinforcement Learning without Safe Constraints vs ILP</td>
          </tr>
          <tr>
              <td><a href="./problem4_single">Problem 4</a></td>
              <td>Reinforcement Learning with Safe Constraints vs ILP</td>
          </tr>
      </tbody>
  </table>

# Multiple Constraints Version

...

# Branch and Version

- Branch: `main`
  This branch should will run on the cuda 13.0+ because reqruirements.txt have the torch 2.10.0+cu130

- Branch: `linux-cuda12.2`
  This branch will run on CPU, torch version is 2.10.0+cpu, and can run on cuda 12.2 but not tested yet. The requirements.txt in this branch have the torch 2.10.0+cpu
