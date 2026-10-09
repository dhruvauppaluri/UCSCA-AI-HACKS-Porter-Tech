# Track Challenges - UCSC AI Hackathon

Archived source snapshot retrieved October 9, 2026. The Google Doc may be updated later; use the source link to check for changes.

Source: https://docs.google.com/document/d/19wl9rK9_BJdIbm7BN4tkNmrui55nEmjhIxGgQClBLSY/edit

---

Track Statements



	Track Challenges
October 10–11, 2026
	Track Challenges
UCSC AI Hackathon @ SF Tech Week
There are three tracks. Pick one for your project. Each track is led by a sponsor who wrote the challenge below. Read the one you plan to build for, including its requirements, so your project is eligible for that track's prizes.
CLOUD AI · CISCO
When AI Agents Disagree
Multi-agent collaboration, adaptation and decisions
	LOCAL AI · ASUS
Build on the ASUS Ascent GX10
Local, private, low-latency AI
	PHYSICAL AI · AMD
Robotics with LeRobot
Robot arms, edge to cloud


CLOUD AI TRACK · CISCO OUTSHIFT | AGNTCY
When AI Agents Disagree
From collaboration to collective intelligence
The challenge
Can you build AI agents that work together, adapt to changing conditions, and make decisions even when they disagree? Individual agents can reason, use tools, and solve complex problems. This challenge asks what happens when independent agents bring different capabilities, incomplete information, competing priorities, or conflicting recommendations to the same problem.
Choose one of three levels. You can build on AGNTCY and its coffeeAGNTCY reference application, or use any other agent harness or multi-agent framework you know. No specific framework, model, or platform is required.
LEVEL 1 · BUILD AN AI TEAM
Build or extend a multi-agent system with specialized agents that collaborate to solve a problem.
Examples: add shipping, quality control, or sustainability agents to a coffee supply chain; a campus assistant for transportation, scheduling, and events; a research team that gathers, analyzes, and verifies information.
Success: distinct capabilities, meaningful communication, and a completed shared task. Explain why multiple agents help and what your team added beyond the starting demo.
	LEVEL 2 · BUILD AN ADAPTIVE AI TEAM
Agents that dynamically discover collaborators, adapt to changing conditions, and resolve conflicting recommendations.
Examples: a coffee exchange finds new suppliers after a shipment fails; a disaster response team reallocates resources; a campus planner adjusts when transportation or facilities become unavailable.
Success: dynamic discovery or capability-based selection; a response to at least one unexpected change or failure; resolved conflicting recommendations; and an explanation of why agents were selected. Running an existing demo alone is not enough.
	LEVEL 3 · BUILD A COLLECTIVE INTELLIGENCE
At least three autonomous agents with competing objectives that deliberate, negotiate, and try to reach consensus without a central decision-maker.
Examples: energy agents balancing cost, sustainability, and reliability; emergency agents negotiating scarce resources; research agents weighing conflicting hypotheses.
Success: genuine disagreement, a real deliberation or negotiation mechanism, the effect of new information, and a comparison against a simpler baseline. Consensus is not always the right outcome; explain when the system should stop, keep the disagreement, or ask a human.
	How your project will stand out
Functionality
	Does the system work and accomplish its objective?
	Innovation
	Is the approach creative or technically interesting?
	Agent intelligence
	Do agents show meaningful reasoning, autonomy, collaboration, or collective decision-making?
	Engineering
	Is the solution well designed, understandable, and technically sound?
	Demonstration
	Can the team clearly explain and demonstrate it?
	Learning and evidence
	What did the team discover, and what evidence supports it?
	What to submit
* A working demonstration
* An architecture overview
* Technical implementation details
* Results and lessons learned
Getting started
You do not need to build a multi-agent system from scratch. AGNTCY is an open-source Linux Foundation project that provides infrastructure for agent discovery, identity, messaging, and observability. coffeeAGNTCY is a practical starting point: its Corto app is a smaller entry point, and Lungo explores richer supply-chain interactions. They illustrate patterns but do not satisfy any level on their own.
Explore the demo, extend it with your own capabilities, experiment with a controlled change or conflict for your level, and demonstrate a repeatable scenario that explains why the agents acted as they did.
FULL CHALLENGE DETAILS: CISCO CLOUD TRACK STATEMENT

PRIZES:
1st Place: 4x Ipad Pros
2nd Place: 4x Nintendo Switches
3rd Place: 4x Razer Viper v3 Pro wireless esports gaming mouse
Resources
github.com/agntcy/coffeeAgntcy
github.com/agntcy/coffeeAgntcy/blob/main/TUTORIAL.md
docs.agntcy.org/coffee-agntcy/get-started
github.com/agntcy/app-sdk
github.com/agntcy
outshift.cisco.com/internet-of-cognition
Cisco is also running a workshop on the coffeeAGNTCY tutorial. Time and room will be posted on Discord.

LOCAL AI TRACK · ASUS
Build on the ASUS Ascent GX10
Fast, private, offline-capable AI where the data lives
Why local AI matters
Building locally removes cloud latency, keeps data private and secure, keeps working offline, and cuts infrastructure costs. Local AI lets developers build faster, more responsive, and more personal experiences right where data is created.
Track overview
Teams in this track must build on the ASUS Ascent GX10, an ultra-small AI supercomputer based on NVIDIA DGX Spark, with the NVIDIA GB10 Grace Blackwell Superchip and 128GB of unified memory. Every GX10 comes pre-loaded with NVIDIA Nemotron.
Eligibility and requirements
* Develop locally on hardware. Build and demo your project running locally on the ASUS Ascent GX10.
* Hardware care. Return all borrowed hardware in the same condition before the closing ceremony.
* Give feedback. Complete a short survey about your experience with ASUS local AI hardware before judging ends.
What ASUS is looking for
* Solve real problems. Address meaningful challenges or deliver impactful experiences.
* Showcase innovation. Creativity, technical originality, and novel use of local hardware acceleration.
* Show the value of local AI. Make clear why running locally matters for your project.
* Inspire and engage. Bring utility, joy, or inspiration to users.
Prizes
1st place: 4x NVIDIA 5080 GPUs
2nd place: 4x ROG gaming keyboards
3rd place: 4x ZenScreen portable monitors
Content challenge
Post about building in the ASUS Local AI track on Instagram, TikTok, LinkedIn, or YouTube for a chance to win a prize (announced at the opening ceremony) and get reposted by official ASUS accounts. Vlogs, project walkthroughs, team videos, and LinkedIn posts all count.
FULL CHALLENGE DETAILS: ASUS LOCAL AI TRACK STATEMENT
Resources
build.nvidia.com/spark

PHYSICAL AI TRACK · AMD
Robotics with LeRobot, edge to cloud
Bring AI into the physical world
This track is built around Hugging Face LeRobot, the open-source library for training and deploying robot control policies, running on AMD hardware from edge to cloud.
Two missions
* Mission 1 · Pick and place. Set up the full LeRobot environment end to end on AMD's stack, then pick up a small object (a LEGO brick, a game piece, a pen) and place it in a bin. Simple to start, and it proves the whole pipeline works.
* Mission 2 · Freestyle build. Define and build your own real-world robotics application using AMD edge (AI PC) and cloud resources.
Hardware
AMD is providing SO-101 robot arm kits for hands-on prototyping, AMD Ryzen™ AI PC for on-device edge inference, and AMD Cloud access with AMD Radeon™ R9700 AI Pro GPUs for model training.
Requirements
Winning teams must publish their full code, training datasets, trained models, and demo videos publicly on GitHub.
Prizes
1st Place: 🥇 The winning team: Compute Credits - $1,000
2nd Place:🥈 The runner-up team: Compute Credits - $500

FULL CHALLENGE DETAILS: AMD PHYSICAL AI TRACK STATEMENT
Resources
wiki.seeedstudio.com/lerobot_so100m_new
github.com/npurusho/AMD_Hackathon
AMD is hosting a pre-event webinar. The link will be posted on Discord.
Co-hosted by hack.dev with the UCSC GenAI Center and Baskin Engineering.
