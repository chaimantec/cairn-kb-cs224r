---
title: "Lecture 11: Model-Based RL"
lecture: 11
video: https://www.youtube.com/watch?v=PvqyGnOirgA
source: "copy-edited from the YouTube auto-captions"
verbatim_original: not committed - regenerate from the video (see below)
material: ../slides/11-model-based-rl.md
---

# Lecture 11: Model-Based RL — transcript

This is the edited transcript of a lecture by Chelsea Finn. The captions have been repunctuated, stripped of filler and false starts, had mis-heard terms restored, and had student turns marked `*[Student question, inaudible]*` or `*[Student answer, inaudible]*` because only the lecturer is audible; no content was added, removed or reordered, and every `[M:SS]` marker is in its original position. Where the captions are garbled and neither the slides nor the context settle the wording, the caption words are kept and an `[Ed: …]` note says so. The verbatim captions are not committed to this repo. They can be regenerated from the video id above with the `cairn-kb` skill's `fetch_transcript.py` piped through `transcript_to_md.py`. Terminology was cross-checked against [the lecture's slides](../slides/11-model-based-rl.md). Caption sound tags (such as a throat-clearing or a snort) were removed. Spoken notation is kept spelled out as it is said ("s t plus 1", "a t plus 1 to a t plus 1 plus h"); the formulas themselves are on the slides and in the wiki page.

## Restorations

| Caption | Restored | Confirmed by |
|---|---|---|
| highle recap [0:05] | high-level recap | context |
| reinforce [0:50] | REINFORCE | context; earlier lectures |
| PO [0:50, 1:35, 2:21] | PPO | context (a policy and a critic, an actor-critic method) |
| important sampling (split by the 1:35 marker) [0:50] | importance sampling | context; the split is left as captioned |
| SACE [1:35, 1:11:21] | SAC | context |
| DQM [2:21] | DQN | context |
| Earl [3:06] | RL | context |
| modelbased, model based | model-based | context |
| index robot manipulation [5:22] | in dexterous robot manipulation | slide 5 |
| V2 and Sora [6:56] | Veo 2 and Sora | slide 6 |
| learn simulator [10:02, 20:10] | learned simulator | context |
| ST [16:12]; ZT, ZTS like ZT plus one [17:02] | s t; z t, z t's, like z t plus 1 | context |
| ps1 given st; ST+1 [19:24] | p of s t plus 1 given s t, with an `[Ed: …]` note; s t plus 1 | context |
| backrop [24:55] | backprop | context |
| RSI [27:13]; trading some reward function [27:13] | r of s, a, with an `[Ed: …]` note; training some reward function | context |
| stoasticity, gausian [28:00] | stochasticity, Gaussian | context |
| gradientfree, zeroth order, cross entropy method, reample [38:03] | gradient-free, zeroth-order, cross-entropy method, resample | context |
| a plus three um or a plus two [41:55]; at and at plus1 [42:42] | a plus three, or a plus two; a t and a t plus 1 | context |
| overp parameterized, over parameterized [43:28, 44:14] | overparameterized | context |
| capital a [45:47] | capital A | context |
| if this is paralyzed [47:19] | if this is parallelized | context |
| action at a at plus1; a t+2; STS [51:13, 52:03] | action a t, a t plus 1; a t plus 2; s t's | context |
| openloop, warmstart, warm start | open-loop, warm-start | context |
| s t +1; a t +1 to t +1 + h [57:28] | s t plus 1; a t plus 1 to a t plus 1 plus h | context |
| something like 110 steps [59:01] | kept as captioned, with an `[Ed: …]` note | spoken only |
| off- policy; comput intensive [1:01:20] | off-policy; compute-intensive | context |
| the next eight time steps [1:04:23] | the next h time steps, with an `[Ed: …]` note | context |
| dextrous [1:06:40] | dexterous | context |
| simulate experiments; soft actor and; pdddm; mpo; pets [1:09:47] | simulated experiments; soft actor-critic, with an `[Ed: …]` note, and; PDDM; MBPO; PETS | slide 34 |
| bouting balls; po poor [1:10:35] | Baoding balls; poor | slides 34, 36 |
| 180°ree [1:12:06] | 180° | context |
| this affects your next action [20:10] | kept as spoken, with an `[Ed: …]` note | context |
| [clears throat], [snorts], >> | removed; student-turn marker inserted where it belongs | n/a |

## Ed notes

Six inline notes mark places where the captions are kept or the restoration rests on context alone: [19:24] captions "ps1 given st" (restored to p of s t plus 1 given s t); [20:10] "this affects your next action" kept as spoken, with the context pointing to the next state; [27:13] captions "RSI" (restored to r of s, a); [59:01] "something like 110 steps" kept as captioned; [1:04:23] captions "eight" for the second mention of the horizon h; [1:09:47] captions "soft actor" (restored to soft actor-critic). At 31:07 Finn defers part of a student's question to a later slide ("in eightish slides or something"), and at 1:06:40 says the second way to handle long horizons will be covered "on Friday"; both are kept as spoken.

**[0:05]** Great. So, we had some feedback in office hours and other times that it might be helpful to have a high-level recap of the algorithms that we've talked about so far to try to understand where some of the pieces fit together. So, I put together one slide that is a little bit more high level but actually captures almost all the algorithms that we've seen so far. And in particular we've talked about both online and offline algorithms, where online refers to scenarios where the policy will be used to collect data, and offline means that we're given a data set and we cannot do any additional data collection in our environment. And in the offline setting we talked about two different classes of

**[0:50]** reinforcement learning methods that use on-policy data collection in order to improve. We referred to the first type as on-policy RL and the second type as off-policy RL, where on-policy RL you can collect data but only from the current policy. You can only use data from the current policy. And this means that you are not going to be reusing past data. You can't use offline data and so forth. And this included algorithms like REINFORCE or the vanilla policy gradient method. And then within the realm of off-policy RL, this was where we could start to reuse data from other policies, including data from policies before we took a gradient step on our policy parameters. And this included algorithms like PPO and importance

**[1:35]** sampling. And then we talked about algorithms that are more off-policy that can reuse data from policies that are even different, even more different, and where we use a replay buffer to store the data from those past policies, and that included algorithms like DQN and SAC. And then I colored these different algorithms different colors because the blue algorithms only involve training a policy and so we refer to them as policy gradient methods. PPO and SAC learn both a policy and a critic and so we refer to them as actor-critic methods, and there's also algorithms that only learn a critic and don't explicitly learn a separate policy, and those are often referred to

**[2:21]** as Q-learning algorithms such as DQN. I should also mention that PPO is often actually referred to as a policy gradient method, although you do actually learn a value function and so we typically are referring to it as actor-critic in this class to avoid confusion. And then separate from these classes of online methods we also have offline RL methods that we started talking about. And these built upon a lot of the ideas we saw in online RL, but I have to cope with the fact that you're not collecting any data at all. And we looked at two different approaches there. One is to constrain to only considering actions in the data. And this included algorithms like AWR, AWAC and IQL, and also an algorithm

**[3:06]** like CQL that implicitly is constraining you to actions in the data, but not so explicitly, and this is through conservatism. Now for these RL algorithms, generally as you move right you need more online data, and so that means that typically they're going to be kind of more expensive, because online data is often more expensive than offline data, because it's harder to repeatedly collect a large amount of online data. And then lastly, before we talked about RL, we also talked about imitation learning algorithms, and this can also fall within the bucket of offline and online, although there's a bit different set of

**[3:51]** assumptions. The first imitation learning algorithm we covered was vanilla behavior cloning, where we're taking a data set and trying to imitate the actions in that data set. And then we also talked about an online algorithm that is online both in terms of the policy and in terms of querying an expert, like asking an expert what is the right action on this new data that I'm collecting. And I put these in a little bit of a separate category because they have different assumptions. You don't need a reward function, but you do need pretty good data, and you're not going to be able to outperform the average behavior in your data set. Cool. So, that's an overview including a lot of these different

**[4:36]** terms that we see in reinforcement learning like offline, online, off-policy, on-policy, and so forth, and imitation learning versus reinforcement learning. Cool. So, what we're going to talk about today is model-based reinforcement learning. And model-based reinforcement learning can actually be applied to all of the different classes of reinforcement learning methods on the previous slide. Someone has a special place in my heart because I used to do actually quite a bit of work on model-based RL and I think that there's a lot of really cool things that you can do with model-based reinforcement learning. We'll talk about the key idea behind it, how to learn models, how to use those models, including with planning and data generation. And then we'll also

**[5:22]** look at one case study in dexterous robot manipulation. And as a teaser for that, the methods that we'll be talking about today will allow a robot to do something like this, which is pretty cool. This hand has more than 20 degrees of freedom, more than 20 different joints that it's controlling. And then the key outcomes are trying to understand how we can learn and use dynamics models and the different challenges and trade-offs with this whole class of methods. Cool. So the key idea behind model-based reinforcement learning is that essentially we're going to try to learn a simulator. Can we learn how to predict the future for actions that we might be taking? And for robotics and driving and

**[6:11]** other physical systems this might be modeling the physics of some particular system, if you can directly observe the underlying physical entities in that system. And alternatively, if you can't directly observe the different entities in that physical system and model them, then this might look something more like video prediction conditioned on actions that the agent wants to take. So for example, I might want to predict if I say something to someone else, how will they react? Or if I for example move my legs in a certain way, will I fall down? Will I go that way? Will I go that way? And so forth. In terms of what this video

**[6:56]** prediction might actually look like, we've seen pretty cool videos coming out of large models like Veo 2 and Sora. And these are examples of generated images, videos that are generated by a neural network. And you can imagine that these kinds of models could be very useful for training systems like cars, robots, quadrotors that are flying through different places in order to learn about how their actions affect the world. Now models could look something like models of the physical world. They could also look at models of other things. So for example, if you wanted to learn a policy to make money on the stock market, then one of these simulators

**[7:42]** would be trying to predict what will happen to the stock market over time and if your actions have any effect on that. And then if you're trying to train an agent to play a game, then this sort of model will underscore the rules of that game. And actually in a lot of games you might already know the rules of the game and you may not even need to learn a model in the first place. For example in chess, if you take a particular action you know the outcome, you know how the piece will move as a result of that action. And then it comes down to, in two-player games or three-player games or multiplayer games, you might need to model other players. Modeling other players, you can sometimes consider that as part of the dynamics but oftentimes it's actually modeled separately. So you

**[8:29]** don't necessarily need to consider it as part of this simulator of the world, and for example you might be able to use the model of yourself as a model of another agent. Okay, so if we can learn a simulator, how would we use it? So the sketch of an algorithm would be something like this, where we get some data. Then we learn a simulator on that data that's going to be trying to predict the future for the present and some action that I might take from the present, and then try to run your favorite method inside the simulator. So for example, maybe you have a

**[9:15]** simple scenario where you want to get from the green point to the red point. You might collect some data. This is collected from some policy. Then you would train some neural network conditioned on a state and an action: what is the next state? And one way that you could learn this is just by minimizing the L2 loss between your predicted next state and the next state observed in the data. And then in terms of running your favorite RL method, you could use one of the RL methods that we've covered, or you could use a planning method. And we'll talk about a couple planning methods today. Cool. So now I have a question for you. So here's a sketch of an algorithm.

**[10:02]** Does anyone have thoughts on what might go wrong if you follow this sketch? Yeah.

*[Student answer, inaudible]*

So the idea there was that your simulator might not always be accurate, and so similar to what we saw in reward modeling and also arguably what we saw when training Q-functions, you might exploit those inaccuracies. Those inaccuracies might compound and then what you've learned in your learned simulator might not transfer to the real scenario that you're trying to learn a good policy in. Yeah.

*[Student answer, inaudible]*

So there's also this data coverage issue. So similar to what we've seen when training Q-functions and other things, if this pi zero, if this initial policy is different from the policy that you're ultimately

**[10:47]** learning, then you're going to have this mismatch and the model might not be accurate for the states that are visited by the policy that you're learning. Cool. So this is basically what's on the slide, exactly. So if you have some environment, maybe you want to navigate a maze, then you might collect some data set from some policy. Maybe the policy does something like this yellow arrow. This is your data set. But this data set doesn't cover the whole maze. And so then if you wanted to go to this part of the maze, for example, or this part of the maze, then your model probably wouldn't be very accurate about those parts of the environment. And so then

**[11:33]** you wouldn't be able to learn a policy for reaching those parts of the maze. And then for the second point, you might not always have an accurate simulator. So here's an example of a video generated by Sora with the prompt, a humanoid robot standing near a table with a red, green, and blue cubes on it performs a cube stacking task with red on the bottom and blue on the top. And you can see that the video generation does some creative things with physics, where it changes the colors of the block and synthesizes blocks out of thin air. I would love for my robot to be able to do something like this. Seems very useful. But this is not modeling what robots can actually do in reality. And to this point, when you're

**[12:19]** actually trying to develop this kind of model, oftentimes you might need to make choices that are dependent on the domain. Like if you're trying to train a video prediction model versus training a model of low-level physics with friction or training a model of the stock market, you might have some choices to make in terms of what are the inputs to that model and how are you training it, what's the architecture and so forth. And some domains are going to be much easier to simulate than others. And then lastly for this last part, even if you have a really good simulator, this doesn't mean that it'll be easy to get a good policy or to get actions that are accurate. As a result, you still have to do the RL process,

**[13:05]** using the simulator. And you need to account for inaccuracies in the model during this process, because usually your model is not perfect. You don't have a perfect simulator of the world. That's the key idea behind model-based RL. Now let's talk a little briefly about how to learn a dynamics model. We don't have too much time to cover this so we'll just talk about it briefly and then we'll spend most of the time on how to use a dynamics model. Okay. So in terms of learning a good simulator or a good dynamics model, there are some scenarios where you might know it already and you could just use it, and then you're in great shape. There's also

**[13:51]** scenarios where you might actually know most of the model. Like if you are in a simplified robotic control setting where you're trying to push an object on a table, maybe you know most of it, but you don't know the friction of the table, the friction coefficient of the table. Or in driving, maybe you have a pretty good model of how the car is going to accelerate, like how it will move forward with a given steering command and a given gearing, and you just need to model some additional things. In those cases you might actually be able to leverage knowledge that you know about the world and then fit those unknown parameters using the data.

**[14:36]** This is actually I think one of the cooler ways to try to incorporate domain-specific knowledge into reinforcement learning algorithms. And then in almost all scenarios in practice you don't really know the model. And this might be because, for example, if you're trying to manipulate some objects, maybe you roughly know the laws of physics, but you don't know the vast majority of how an object is going to behave because it has lots of different properties and there's many different objects. Or you don't know how people are going to interact in the world or other drivers are going to behave in the world and so forth. And in these scenarios the typical practice is to learn the model from scratch with a neural network or

**[15:23]** some other function approximator. Now there are a couple options here as well. So if you are using image observations for example and you're trying to predict, if your state space for example is an image of, I don't know, maybe you have a car and there's other cars here and you're trying to predict, in video, where those cars will be moving in the future. And maybe there's I don't know some trash on the ground or something like that. One option is to

**[16:12]** I'll refer to this image as s t. One option is to generate an entire future video like the videos that we saw on the previous slide. This would be conditioned on actions. But this might be fairly expensive to do, because you have to basically be predicting this entire video over time. That's a pretty large thing and you have to model all the different details of how all of the pixels are going to change in the future. And so one alternative option that is sometimes used to learn dynamics models is that instead of modeling in the image space you could instead learn some

**[17:02]** lower-dimensional representation and then model in that representation space. So you for example could learn, for a given state, some sort of encoder that maps this to a lower-dimensional representation space, maybe we'll refer to this as z t, and then instead of learning a model in the original image space we could then try to learn a model that predicts future z t's, like z t plus 1. This can be a good alternative in scenarios where it seems like it'll be very expensive to learn a full model. It also has its own challenges because you need to figure out how to get this representation in the first place.

**[17:49]** And if you have a good way to get good representations it can be a good choice. Yeah.

*[Student question, inaudible]*

So the question was, we are going to be losing information by going to a lower dimension, and so is the main reason to do this in terms of cost? And yeah, exactly. So the main reason to do this is to try to reduce some of the computational cost associated with this, because you might be losing information. And the key idea is that there are some parts of this observation that don't matter. Like perhaps the exact texture of the road, like whether it's really dark black or maybe a little bit worn down. In many scenarios that might not matter. And so if you can learn a representation that discards that information that's irrelevant, then you don't have to spend so much compute modeling all those details that don't matter.

**[18:36]** The other thing that I'll mention here is that oftentimes when you're trying to learn these models, typically with supervised learning or generative modeling techniques, you will also need to learn a reward model as well. And the reason for this is that if you want to be using your model to generate data or to see into the future, you need to see into the future not just the states but also how those predicted states will lead to rewards. And so we'll talk about what that looks like on the next couple slides.

*[Student question, inaudible]*

The question is does model learning have to happen end to end? The way that it works is we are optimizing for our ability to generate future states. And

**[19:24]** There are actually pros and cons that we'll talk about with regard to that objective. But you can think of it as, one way to think about it is, how well can I model the distribution of p of s t plus 1 given s t [Ed: captions "ps1 given st"]. Another way to think about it, especially if you have more deterministic dynamics, is how well can I minimize the error between my predicted s t plus 1 and the actual s t plus 1. So, how do we go about using a dynamics model? The

**[20:10]** first approach that we'll talk about is we'll actually start by talking about planning. And so this is actually going to be a little bit different than simply running reinforcement learning inside your learned simulator. And the way that this could work is you can think about your MDP where you have your current state that you're in and then you take some action and then this affects your next action [Ed: from the context, the next state] and then as a result of that you'll also get some future reward. And then this process will repeat. You'll take another action that influences the state after that

**[20:57]** which will yield some additional reward. This reward might also depend on the action that you took as well. Now in some of the past reinforcement learning works that we talked about we assumed that we didn't really know this relationship. We don't know how the actions affect future states and so we have to figure out how to optimize rewards without using direct gradients. But if we can actually predict what a future state is and what a reward is based off of that, if we train a reward model and a dynamics model, then we could actually just directly backpropagate from rewards

**[21:45]** into actions. And so we could potentially directly pass gradients from your rewards into your actions. And what this would look like is your objective would be to maximize either just the current action or maybe the next sequence of actions for the sum of future rewards. Maybe we'll do this, t prime, from, for some horizon. So maybe we're optimizing for the rewards from time t to time t

**[22:31]** plus h. So what this would look like specifically, I guess we can show it on the slides as well. So we're trying to maximize the sum of rewards, and we often refer to this kind of problem, looking into the future, as planning a sequence of actions that will maximize rewards. And so if we have some model of how we predict future states and actions and rewards, and if we're modeling that with a neural network, then we can also compute the gradients of that neural network to backpropagate rewards into actions as follows. And so then specifically what this would

**[23:17]** look like is, well, first you would collect some data, then you would train your model. Maybe that's to minimize f of s, a minus the, this is your predicted next state. This is your true next state. And then once you have this model you could then find a set of actions that is actually, just to be explicit, for this optimization we're minimizing the parameters of our model.

**[24:04]** And then the third step is to try to find a set of actions. So this would be a hat t to t plus h. This is what is the course of actions that will lead to good rewards. And this would be the argmax of the sum of rewards. Well, sum of t and this is over

**[24:55]** so writing gradient descent on actions is a little bit like backpropagation to actions, is a little bit different from the typical backprop that we look at. So oftentimes when we run backprop we're doing it with regard to parameters of a neural network. Here we're actually backpropagating into these actions right here. And so specifically what this third step would look like is you would need to basically initialize, just like you randomly initialize your neural network parameters, you'll need to initialize your actions. Then you need to compute your gradient.

**[25:41]** So this would be the gradient of the sum of rewards with respect to the actions that you take. And to compute this gradient, you can think of it essentially like if you use the chain rule, the gradient will be the gradient of the sum of rewards with respect to the states. So that would just be based off of your reward model. And then if you apply the chain rule you could then have a second term that's d states d actions to actually get the

**[26:28]** gradient of the rewards with respect to the actions. And then once you have your gradient then you could use it to update your actions with gradient descent. And so then maybe this is your initial a hat. A hat t to t plus h is going to be your previous a hat minus alpha times this gradient right here. I'm going to put this separately.

*[Student question, inaudible]*

Yeah, great question. Actions would be

**[27:13]** continuous in this setting. Yeah. We'll also look at scenarios where we can do it with discrete actions as well.

*[Student question, inaudible]*

We're assuming rewards are deterministic, but we've actually always assumed rewards are deterministic. And you can consider any sort of non-determinism being encapsulated by the dynamics and in the state variable.

*[Student question, inaudible]*

Correct. Yeah. So here we have a dynamics model and I guess I didn't explicitly write this but you would also be training some reward function. Maybe this is r of s, a [Ed: captions "RSI"] for example. And this would be using some of the reward learning techniques that we've talked about. There's also some scenarios where you might know the reward. So it might be for example forward velocity if you're trying to learn how to walk, or some distance

**[28:00]** to some goal, that sort of thing. And in those cases you don't need to learn a reward function. Yeah.

*[Student question, inaudible]*

So the question is if your dynamics model is stochastic, how do you backprop it? That gets into a lot of, it depends on how you're modeling the stochasticity. So if you're using a Gaussian for example there are certain tricks for differentiating through samples from that Gaussian, whereas if you're using something like a diffusion model it would look a bit different. So it is model dependent, and so far we're talking about deterministic models. Yeah.

*[Student question, inaudible]*

So the question is will this

**[28:46]** start to exploit errors in the reward? Yeah. So this will start to exploit both errors in the reward but also errors in the dynamics model. So if your dynamics model is inaccurate or if your reward model is inaccurate, if either one of them is inaccurate, then you might get actions that aren't actually going to take good actions in the world. And that's a problem. And so we'll talk about that in a second. Yeah.

*[Student question, inaudible]*

So this is a good question. So, I guess instead of repeating it, I think what would be helpful is we aren't actually learning a policy here. And that's essentially what you're pointing out. And this is now becoming our policies, quote unquote. This is

**[29:32]** our behavior and, did I have a good term for this? I guess I mean yeah, this is sort of the policy in quotes, and this is what we call planning, where at every single state we're actually going to be planning forward and thinking ahead, like what are the actions I should consider that I think will lead to good reward, and so we'll actually be doing this optimization process at every single state when we're doing planning. So on this slide, we're going to be running some policy to collect data, learning a model to try to be able to accurately model the dynamics, potentially also learning a reward

**[30:19]** function as well. And then we'll be backpropagating through the reward and the dynamics in order to choose actions. And then once we choose those actions we'll be executing these actions in the environment. And then after we execute those actions we'll be rerunning this planning process again. So in particular I guess maybe to explicitly write that out we'll have a step four here that is to actually execute a hat t to t plus h. And then you will move on

**[31:07]** back to step three after you've executed these h actions. Yeah.

*[Student question, inaudible]*

So the question is we are only planning over a horizon of h, and what if we have an infinite horizon? In principle, the math, well it's a good question. We'll talk a little bit about this. For now we're going to be only considering this short horizon planning. And in eightish slides or something I'll talk a little bit more about what happens if you want to do infinite or very long horizon things. And we'll also talk about some of the shortcomings of this. Yeah.

**[31:54]** Okay. Now, so far we've talked about gradient descent for optimizing actions. And this is actually not the only way that you can do optimization. So it's probably the optimization technique that you're most familiar with because we use it all the time for neural networks. But there's other ways that we can optimize a function, specifically when we're trying to optimize the sum of rewards. And if we think about gradient-based optimization, typically we have something we're trying to optimize over, in this case it would be we're trying to be optimizing over actions, and the thing that we're trying to optimize is the sum of rewards

**[32:40]** maybe. Yeah. And so actually also there's a small mistake here. There should be a plus sign because we're maximizing not minimizing. So the way that gradient descent works is you randomly initialize some action like right here and then you evaluate the gradient at that position. So maybe you find that the gradient is pointed this way. And then you say okay I should move probably over here. And then maybe you sample this. You choose this action because that's a step size away. You evaluate the gradient there and you find that the gradient maybe looks something like this. And then you say okay maybe I should move further to left but a little bit less because the slope is less steep. And then you get

**[33:27]** maybe something that looks like this. And you're like okay the gradient is zero and so maybe I won't be sampling any further actions. And then your final choice after running gradient descent is this value right here. Now an alternative way to try to do optimization is instead what I'll call sampling-based optimization. This is also called gradient-free optimization. And the way that this works is if you have some set of actions and some sum of rewards, instead of just starting from one place and evaluating the gradient, we're actually not going to use the gradients at all and we'll just sample a bunch of actions and see what the values are. So

**[34:13]** maybe we'll for example sample this action. We'll sample this action. Maybe sample this one. Maybe sample this one. And so forth. And the first version of a sampling-based method could just be to sample a bunch of actions, a bunch of things in parallel, and then pick the best one. This is sometimes basically just called random sampling. Sometimes it actually works better than you think it might. And if we were to do that then we would say that okay this action has the highest and so maybe I will return the corresponding action for this value. But you can also do something a bit

**[34:58]** better than that even without gradient optimization, which is that you could look at these samples and say okay it looks like maybe we have one additional sample that we've maybe also sampled something like this. And in that case, one thing that we could do instead of just considering this one and taking the best is we could say maybe we should sample more over here because these are the ones that were the best. And so we could basically try to sample more from the things that were good. So we could take the ones that were the best, fit some kind of distribution over those, like a simple distribution like a Gaussian, say that okay these were the actions that were the best. So maybe we could fit some Gaussian or sample more

**[35:44]** from the actions here and then do another round of sampling from that new distribution, where now you're going to be sampling more from this distribution. You find that when you sample for a second round maybe you get samples that look like this. And then that allows you to hone in on what a good action might be. So on the slides I did this instead of with actions and maximizing rewards, I did it with parameters and minimizing some loss function. But all the ideas are the same. So for gradient descent we pick an initial value, evaluate the gradient, move in the direction of the gradient and

**[36:31]** repeat that process. There's some underlying function that we don't know, and so we're trying to get to the minimum of that function using these gradients. And likewise with sampling, or another way to call this is zeroth-order optimization, we're instead going to be sampling points and trying to figure out what a good value is just from sampling. And again there is some underlying curve here that we don't know. We can only evaluate samples or, in the case of gradient descent, the gradient of those samples. One other nice thing about sampling-based optimization is that it means that you don't need gradients. So it can apply to discrete variables. It can also apply to scenarios where you maybe know

**[37:18]** the reward function, you know you have reward values, but you don't know what the gradient of the reward function is. Like if it's a discrete reward function then getting the gradient might be challenging. Okay. And then to formalize this iterative sampling process where we're trying to sample more from the good actions, one method for doing this is the cross-entropy method. There's also a way to do this, well if you have discrete actions you could also just, well there's somewhat of a different process, but you could also just use random sampling if your action space is small enough. So what the cross-entropy method does, not to be confused

**[38:03]** with the cross-entropy loss, is you sample from some initial distribution. This could be a uniform distribution. Then rank the samples according to their loss, to get, like, this would be the best one. This would be the second best, this third best, fourth best and so forth. And then take the best ones, like these three ones, and then fit a distribution, like a Gaussian distribution, to those best samples and then resample from your distribution, rerank and repeat this process. So it's basically a very simple sampling-based way to try to get, in this case, good actions. It's a general optimization technique

**[38:50]** that could be used for any sort of objective. Now, cool, and then after this process, eventually after you run some number of iterations, you'll return the best sample that you found in the last iteration.

*[Student question, inaudible]*

The question is, can you also use sampling to approximate gradients? Yes. So there is something called finite differences, where if you don't have a way to get gradients then you could try to sample points very close together and then look at the difference between this point and this point and use that to estimate the gradient. There are pros and cons to it but it's definitely a valid approach. Yeah.

*[Student question, inaudible]*

The reason the

**[39:35]** question is why do I put the quotes around the policy here? The reason is that we're not learning a parametric policy. We're not learning a neural network to represent a policy. And there's actually an optimization process underlying the policy here. And so typically it's a policy in the sense that it is mapping from a state to an action or a state to actually multiple future actions. But it's often not referred to as a policy, to differentiate it from policies that are more like neural networks, that are more compact. Well, you can also have a linear function that's mapping states to actions. The main thing is that

**[40:21]** it's a process. It's an iterative optimization process rather than some sort of closed form, like here's an action for a state. Yeah.

*[Student question, inaudible]*

So the question is, can we talk a little bit about how this policy is stored? So the only thing that we have learned at this point is a dynamics model and potentially a reward function. And so the only thing that we have stored is the parameters of the model and the parameters of your reward model. And then when you're actually trying to figure out what to do and what actions to take, what policy to follow, we don't have a neural network that's telling us what actions to take. Instead, every single time step, you're thinking, okay, what's going to happen if I do this, this, or this? And then

**[41:08]** choosing actions that you think are good as a result of that thinking process. This ends up being that you're using a lot more compute as part of this process and you're doing that in real time as you're trying to figure out what actions to take. Yeah.

*[Student question, inaudible]*

So we're not recording what we're doing. Yeah. Right now. Yeah. Yeah.

*[Student question, inaudible]*

So the question is, when we sample actions, can some of them be out of distribution? So yeah, when you select the distribution to sample from, your actions might be out of distribution compared to the data that the model was trained on. And we'll talk in the next slide on how to deal with that. Yeah. Yeah.

*[Student question, inaudible]*

So the question is, because we're doing so much compute at test time, if

**[41:55]** you finished computing and by the time you finish computing the state has changed, how do you account for that? Typically I don't know if I have a great answer for you. I think that typically when people run this in practice they just try to make that compute process as fast as possible so that they don't have to account for the change. I guess one thing you can do is you're planning actually a sequence of actions here. And so if you can estimate how long that computation took, if it took two time steps for example, then instead of starting by executing a, you could actually start by executing a plus three, or a plus two

**[42:42]** depending on how the off by one works. And so that's accounting for the fact that time has moved forward. The downside there is that a t and a t plus 1 might be different from what was actually executed and so you might be in a different state from where you thought you ended up in. And so that introduces challenges. And it's actually somewhat of an open research problem on how best to handle that sort of thing, especially if you're basically running this process asynchronously and also the world is evolving as well. Okay. So in terms of, so going back to

**[43:28]** doing this with gradient descent versus with samples, there are pros and cons to gradient-based and sampling-based, and this helps elucidate why sampling-based actually can be a good choice for planning. So gradient descent is very scalable even to very high dimensions. And it works well especially in overparameterized regimes, like when you have a big neural network that's representing a function that may actually have fewer free parameters. The downside is it requires a nice optimization landscape where your gradients are fairly smooth rather than not smooth, I guess. With neural networks it ends up being, especially because neural networks have high-dimensional parameter

**[44:14]** spaces and are overparameterized, the pros end up heavily outweighing the cons here. Some of the upsides with sampling-based approaches is that it's really easy to parallelize because you can estimate multiple samples in parallel. Whereas gradient descent inevitably has to be a very sequential process where you compute one gradient and then apply it and then compute the next gradient and so forth. And this parallelization can actually in some cases make it faster. It doesn't require gradients if you don't have them. The downside is that it scales poorly to high dimensions because if you have a really high-dimensional space, it's very hard to cover that high-dimensional space with lots of different samples. So in one dimension something like this can actually work really really well. But if you can

**[45:01]** imagine, instead of this being one-dimensional or two-dimensional or 10-dimensional, you have to have your samples cover that space well in order to find a good optimum. Cool. So specifically what the planning approach would look like with a sampling-based optimization is you will still run some policy and train some model like we did in the first two steps and we'll also be choosing actions that are trying to maximize rewards. So essentially things are the same except the way that we do this optimization problem is going to change

**[45:47]** where instead of running gradient descent we could instead sample a bunch of different actions, imagine what will happen in the future and what rewards we'll get, and then potentially refine our selection of actions iteratively to then get a good outcome. And so in terms of what this looks like, if we use capital A as shorthand to denote a t to a t plus h, then version one is if we just do one round of sampling, then we'll sample a large number of possible action choices we could take from some distribution

**[46:34]** then choose the set of actions that had the highest reward. And then run those actions in the environment. The technical name for this is often called random shooting because you're shooting out different possible samples of what could happen in the future, imagining the future, evaluating the reward, and then just picking the one shot that had the highest reward. Now the downside and the thing that's unsatisfying about this first algorithm is that this distribution may not be very good. And so what we could do is we could use the cross-entropy method where we're

**[47:19]** iteratively improving that distribution, where we start from some distribution, maybe a uniform distribution, evaluate, rank and pick the best samples that have the highest rewards, and then refit a new distribution to those best samples and repeat that process, where we're then going to be resampling from that better distribution and then evaluating those samples and repeating. If this is parallelized, this process actually can be quite fast. If you have low-dimensional actions, it's actually also pretty simple to implement. But like we talked about, it doesn't scale well to high dimensions. And so it's best if you

**[48:05]** have a somewhat low-dimensional space that you're considering. This includes both h and a because the optimization that you're considering is over this vector right here. And the size of this is the dimensionality of your action space times the horizon. Cool. Let's run through an example for roughly how this would work. So let's say that we are in this simple beautiful environment and you are trying to maximize reward. Maybe you're trying to get to the highest point because at the highest point you get the

**[48:50]** nicest view. So the reward is just the height of where you get to. You run some policy, you collect some data like this. And maybe when you train a model on this data you learn that when you move right then you go higher. Because generally when you move right the elevation became higher. And then when you try to choose actions according to your model, then you're going to learn that you should go right. So you'll learn that when you're finding actions that maximize reward, your set of actions will be to go right. And then you'll actually execute

**[49:38]** those actions and you'll go right. Which is good for a bit. You'll be maximizing reward for a while and then at some point you'll fall off the cliff and that's not so good. And this is because of the thing that we talked about at the very beginning, there's this data distribution mismatch where the data that you collected at the beginning doesn't cover the data that you are taking with your final policy. So we have a simple fix to this. The simple fix is that when we execute actions in our environment, we can add those actions to our data set and refit our model to that data set, learn

**[50:24]** a more accurate model, and repeat this process. And so then this is starting to look a little bit more like a reinforcement learning algorithm because we're iteratively collecting data, fitting a better model, in this case a dynamics model, sampling actions, choosing actions to execute, and running those actions in the environment. And so then when we do this, we will execute the actions that we planned. So we'll go off the cliff. Then our model will become more accurate. We'll learn that if we go right past this point, we get a lot of negative rewards or we get very low elevation which corresponds to negative rewards. And then we'll be able to learn a model that goes to the top and stops.

**[51:13]** Cool. Now there's a number of things that we could still improve about this algorithm. And let's talk about a few of those. So far we've been executing this set of actions right here in the environment. So we've been planning out a sequence of actions and executing them. And so what that looks like is we have our agent and our agent is running h actions in the environment. So action a t, a t plus 1

**[52:03]** a t plus 2 and so forth in some environment and we're executing those actions without actually having access to the intermediate states. And one alternative that you can consider is something where at each time we're going to be planning a t to a t plus h. But we could also have something where we plan a t to t plus h but we only actually execute a t in the environment. And then we get the next state back in return. And then we replan

**[52:52]** at the next time step a new set of actions. And so the difference between these two things, this is often what's called open-loop, because you're not actually using the feedback from the environment, the intermediate s t's, when you are executing future actions, whereas this is often called closed loop because you're actually replanning as you get new states. So here's what that looks like. Now I'm curious, does anyone have thoughts on

**[53:37]** scenarios in which it might be bad to be open-loop like this, where you're only planning based off of s t and not planning based off of all of the future s t's. Yeah.

*[Student answer, inaudible]*

So when the environment is changing a lot. So if the environment changed in a way that is different from what you anticipated because you maybe weren't able to predict everything that would happen, then you won't react to those changes when you act in an open-loop way. And instead if you're closed loop and you're reconsidering based off of the latest states, then you're able to actually react to those changes and potentially in some cases have a much much better policy. Whereas

**[54:25]** in scenarios where the world is a little bit more deterministic, it can be okay to be a little bit open-loop at times, but you usually never want H to be too large, otherwise you're executing too open-loop. Now we talked about how we can make planning closed loop, which is that instead of executing the entire action sequence you can just execute only the first planned action. So, we're still going to plan a sequence of actions because we need to consider not just the first action, but how all of our actions are going to affect future rewards. But then we'll execute only the first one, observe the resulting state, and then we will replan an

**[55:11]** action sequence starting at that new next state. And then we can also, like before, add the data to our data set and periodically refit our model based off of the latest data. This algorithm right here is something called model predictive control. It's very very common in certain robotic systems, very widely used. It's also used in other simple control systems as well where you can model lower-dimensional systems. It can be quite

**[55:56]** effective for legged robots as well, where you're trying to learn models and you're able to plan over the next couple seconds to figure out what will lead to good performance over the next couple seconds. And this sort of replanning, when you actually are considering a new plan at the next time step, if something changes or if you made a mistake, you slipped or something, this replanning allows you to consider a completely different course of action. And, for example, correct not just for changing environments, it can

**[56:42]** also correct for model errors. And so generally this sort of approach, if your model is accurate in a large part of the state space but in some parts of the state space it's inaccurate, this sort of approach ends up being a lot more robust to those kinds of model errors.

*[Student question, inaudible]*

Yeah, that's a good question. So if there are multiple correct paths, the question is will that multimodality lead to oscillation between those different paths. It certainly can lead to that if you're replanning all the time and you might start going in one direction and then maybe say oh maybe I'll go in a different direction. Sometimes you also won't see that if the state indicates that you've clearly committed to one direction. But so sometimes you see it, sometimes

**[57:28]** you don't. The other way to accommodate this and also address computational challenges is that you can warm-start your plan for any given time step to basically the plan from the previous time step. And so if you had planned a t to a t plus h and at the next time step you are planning based off of s t plus 1, you're planning a t plus 1 to a t plus 1 plus h. Then what you can do is you can warm-start this using everything but the first time step of this previous plan and then randomly initialize the last action and you'll already have some kind of

**[58:15]** initial plan that is sensible and so you don't have to run as long of an optimization procedure and it also helps you commit to the previous action in the previous. Yeah.

*[Student question, inaudible]*

Yeah, you can instead of randomly initializing, for example in our sampling-based thing, you could, yeah, either here or in a sampling-based approach, you could initialize close to the previous plan. You could also add some noise or add some variance to your sampling distribution so that you're not only considering that particular plan. Cool. So replanning helps with nondeterministic environments. It also is really really useful for model errors. It is also compute-intensive.

**[59:01]** So the warm-starting can be helpful if you're actually replanning at every single time step. It means that every single time step you're reconsidering what action to take and so it also can introduce additional compute challenges as well. Sometimes you have something in the middle where you'll have some planning horizon, but then maybe you'll plan not every one step but every three steps or every 10 steps or something like that, where maybe your horizon is something like 110 steps [Ed: number as captioned]. Yeah.

*[Student question, inaudible]*

So it's an interesting question. So the question was, as you're learning and getting more data, presumably your model is going to get better and you have more confidence in your model and so maybe in so far as this is really useful for correcting for model errors you might be able to replan less frequently when you have a

**[59:47]** better model. So that is certainly very viable and usually at some point you will have some kind of final model that you're using after the learning process and you could reoptimize, you could use different hyperparameters with regard to replanning for that final policy versus during the learning process. And during the learning process it can also be valuable, the model errors also help you explore to some extent, because you're going to be exploring exactly the regions where the model is inaccurate because you'll be incentivized to exploit any inaccuracies. Yeah.

*[Student question, inaudible]*

So the question is, it seems like this is an off-policy algorithm, and how do you choose that policy? So yes this is an off-policy algorithm. We

**[1:00:33]** don't need to be using data from our current policy. You can actually use a pretty broad policy and it makes sense to use a pretty broad policy because then you'll get a model that's accurate in more places, assuming that your model is expressive and can fit that broader data distribution. Okay, so far we've covered planning with models. So we plan a sequence of actions. We can do this either with gradient descent or sampling. We then update the model using data collected through this planning process and we can replan to help account for mistakes. This is a pretty simple way to use a learned model. It's also one of the things that I think is really cool about this, if you have a model that's

**[1:01:20]** quite good, you can actually plug in different rewards or different goals at test time even and get a policy that can follow, for example if you have a model that's universally good and you have a robot that's trying to walk, then with that same exact model you can use this process at test time and say I want you to run forwards, I want you to walk backwards, I want you to crouch. And this kind of model is quite task agnostic. The downside is it's compute-intensive at test time and so you need to account for this, and if you have an expensive model and expensive planning process this kind of planning at test time might not be suitable.

**[1:02:06]** And the last thing is we've been talking about this for short horizon problems. This is not applicable to long horizon problems unless you have a shaped reward function. So if you have a shaped reward function that is telling you a lot about what a good state is versus what a bad state is. For example, maybe you have an infinite horizon problem that you want to walk forward forever. You want to learn a policy that can walk forward forever, but you have a very dense reward function that's like forward velocity. Then you can plan for a short horizon in a way that approximates the long horizon problem. But if you have a long horizon problem that doesn't have a very dense reward signal, then this is not a good approach.

**[1:02:51]** Cool. And part of the reason for this is the accuracy of the model. Another part of this is that it's going to be expensive to be planning out a very long horizon sequence of actions because you're not learning a policy in this case. Okay. Now, the question is can we train a policy using this model? And the first option here is you could actually distill a planner's actions into a policy. So you could store these actions and then train a policy to match these actions. This would address the

**[1:03:36]** compute-intensive problem and this would make it a lot cheaper because then you just need to run a forward pass through the neural network to get your actions. This doesn't address the horizon problem though because we're still going to be using this procedure to optimize for actions. And then in terms of using models for longer horizon problems there's two main solutions. The first solution is that right now we're optimizing for a sum of rewards from t to t plus h. And if we want to optimize for a longer horizon of rewards, so far the way that we would have to do that is by optimizing over a longer horizon

**[1:04:23]** of actions, which gets higher and higher dimensional. But another thing that you could do is that if you have a value function or an estimate of a value function, then you could actually use that value function to help this process, where instead of trying to maximize the sum of rewards for the next h time steps, you could optimize for the sum of rewards for the next h time steps [Ed: captions "eight"] plus your estimate of the value of the state that you end up at after taking those h actions. And so this would end up with an algorithm where you're learning both a model and a value function. And then you would be running this planning process. You'd be

**[1:05:09]** considering a bunch of different actions over some horizon. And then past that horizon, you're just asking your value function, is that a good state or a bad state? And so this is a valid way to combine planning with value-based RL in order to select actions. Yeah.

*[Student question, inaudible]*

So the question is, why use the model? Why not just use a Q-function for example? This is why I didn't want to spend too much time on this method, is that in this case you are learning a value function, and you have to learn a value function and a model, you're learning a lot of different things and there's a lot of moving pieces. There is still some merit to using this and it for example

**[1:05:55]** if you know the model then you're going to get a better estimate of future rewards than your value function. And so you can actually use this to plan out future actions. Like in games, it's actually really common to do these rollouts through your model of the game, to the extent that it's computationally feasible, and then use your value function just because it's going to be more accurate. It gives you a better estimate of how good your action is. But it's only if you have a really good model or if it's practical and not too difficult to learn a good model. And then the other approach which I'm thinking

**[1:06:40]** we don't have too much time and so I think I might cover this second approach on Friday and then go through the case study. So let's talk about this case study. So this case study is actually going to be an example of using planning for dexterous manipulation. And the reason why I want to cover this case study briefly is that in my mind it is actually still one of the most impressive robotics results with a five-fingered hand. And it also has

**[1:07:26]** some pretty nice comparisons that are elucidating about what is important for good performance in model-based RL and planning. So the way it works is the states are the position of the hand and the position of objects. This assumes that you know the 3D position of an object, which you can estimate with a computer vision system or put motion capture markers on the objects. Whereas the hand position you can use sensors in the hand itself in order to measure the hand positions. The action space is to control a five-fingered hand. It has 24 different joints that you can control. And the reward is to target some kind of object trajectory where there's also some

**[1:08:12]** penalty for dropping the object. The model is going to be actually pretty simple. The model is going to be a neural network with two hidden layers of size 500. But it's also going to train an ensemble of neural networks. And this is an example that I wanted to look at because this ensembling approach can also help with model errors. And the way this works is instead of just training one neural network, you'll train three neural networks with different random initializations and different mini batch sampling, just with different random seeds. And then if one model is very optimistic about what will happen, that can be counterbalanced by errors in other models. And it essentially gives you a way to

**[1:09:01]** deal with the fact that models will have errors in different ways. So you're not going to exploit the errors in any one model. And then the planner is a modified version of the cross-entropy method. So it's going to be using iterative resampling of the actions and the modifications are that it's going to use a softer reward-weighted mean and it does some sort of temporal smoothing on the actions to encourage and search over actions that are taking smoother trajectories. But you can basically think of it as the cross-entropy method or this kind of iterative sampling-based optimization. And then they alternate

**[1:09:47]** between, in step one, collecting 30 trajectories and updating the model. Okay. So then in the simulated experiments we are going to see a comparison to some model-free methods, methods that are not using a dynamics model. We'll see soft actor-critic [Ed: captions "soft actor"] and an algorithm called natural policy gradient, which is a policy gradient method, and then we'll also see some model-based methods. So the proposed method is called PDDM, we'll also see an algorithm called MBPO, a different kind of planner called PETS, and then also a method that doesn't use ensembles and doesn't use iterative resampling. And here are some videos of what you're able to do with this method. So

**[1:10:35]** we're running planning in the loop of actual execution and we see that it's able to learn how to rotate the Baoding balls in the hand and also write digits on this piece of paper. And we see that if we plot the reward, where higher is better, over the number of data points that are collected in order to fit the model or the policy in the case of model-free methods, we see that first for this really complex task only the proposed method is able to solve it because it's really challenging. And we also see that some of the methods that for example don't do iterative resampling, that for example just do random sampling, shown in yellow, get very poor

**[1:11:21]** performance because you're not able to get to a good part of the distribution and hone in on good actions. And in the case of handwriting, which is a little bit easier, we see that some methods are able to get good performance, like SAC and natural policy gradient get high performance, but they need far far more data than this model-based method which is shown in green, and so the model-based method is using maybe on the order of 100,000 time steps of data whereas the model-free methods are using like half a million time steps. So it's a lot more efficient than model-free methods. It's also more performant than other model-based methods. We also see that the model needs to be sufficiently large. You also

**[1:12:06]** need this ensemble for good performance. And there's different trade-offs with regard to the planning horizon. So you don't want to plan for too long into the future because that's too high dimensional. And then they also did it on a real hand using the method that we talked about. Data efficiency is really important for fragile hardware. This robot hand is very fragile. There actually is one at Stanford in the Gates building and I think all the students are afraid to touch it because they don't want to break it. And then they have to figure out how to fix it because it's also really complicated. And the success rate for 90° turns is actually around 100% after just four hours of data. For 180° turns, which is a bit harder, it's the

**[1:12:52]** success rate is a bit lower, but you're still able to make significant performance with four hours of data. They also had the, they reset the system when it dropped a ball using another robot. Look like this setup. Cool. That's it for today. Next time we'll cover the last part about model-based RL and RL with multiple tasks and goals.
