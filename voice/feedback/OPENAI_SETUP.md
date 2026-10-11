# OpenAI adjustment integration — placeholder

**Deferred / theoretical for the demo:** a live API attempt returned `credit_balance_exhausted`. No successful live adjustment has been verified. Use the original five ElevenLabs client tools and call `propose` directly. The instructions below are for a future activation when API credits are available, not a currently working feature.

The bridge now registers a sixth optional CLIENT tool, `request_adjustment`, and can expose the same function as an authenticated webhook. Both use the same controller and completed run IDs. Do not create another Controller process for the same state folder.

## Install and configure

From the repo, in the environment running the voice bridge:

```bash
pip install openai 'elevenlabs[pyaudio]' numpy
export OPENAI_API_KEY='your-openai-api-key'
export OPENAI_MODEL='your-api-model-id'
export ELEVENLABS_API_KEY='your-elevenlabs-key'
export ELEVENLABS_AGENT_ID='your-agent-id'
```

Choose an API model available to your OpenAI project that supports Responses Structured Outputs. `OPENAI_MODEL` is required; desktop Codex model labels are not necessarily API model IDs. Keys belong in environment variables, not Git or tool-body parameters. A Codex desktop login is not used by this adapter.

## Easiest connection: client tool

Add `request_adjustment` in ElevenLabs as a CLIENT tool, Wait for response enabled. Required string parameters:

- `run_id`: exact completed physical run being corrected.
- `feedback`: user's requested correction in their own words.
- `request_id`: unique adjustment request ID; reuse on retries only.

Description: “Ask the OpenAI model to translate negative drawing feedback into bounded planner adjustments. Call after feedback(accepted=false). Returns a prepared revision or a clarification question; never moves the robot.”

Append to your agent instructions: “After recording negative feedback, use request_adjustment to interpret corrections instead of choosing propose parameters yourself. Preserve the user's feedback. If needs_clarification is returned, ask its question. If prepared_not_executed is returned, explain the change and use its revision_id only when the user requests a test.”

Start `python voice/feedback/elevenlabs_bridge.py` (dry-run mode). Use `--physical` only on the configured robot station for actual runs. Adjustment requests require a completed physical run and explicit negative rating; dry runs cannot impersonate completed robot tests.

## Alternative: ElevenLabs server/webhook tool

Set a separate random bearer secret (24+ characters) as `ADJUSTMENT_API_TOKEN`, then:

```bash
python voice/feedback/elevenlabs_bridge.py --adjustment-api
```

The endpoint binds to `127.0.0.1:8780`. Configure an authenticated HTTPS reverse proxy/tunnel to it before ElevenLabs cloud can reach it. No public endpoint/tunnel has been provisioned here. Keep the normal voice client running; it owns the run state. Do not configure both a client and webhook tool with the same name.

Configure the webhook:

- Method: POST
- URL: `https://YOUR-HOST/request-adjustment`
- Secret header: `Authorization: Bearer YOUR_ADJUSTMENT_API_TOKEN`
- JSON body: `run_id`, `feedback`, `request_id` (strings).
- Allow enough timeout for the model call and IK generation (up to roughly a minute with a retry).

Local test, after a physical run has received negative feedback:

```bash
curl http://127.0.0.1:8780/request-adjustment \
  -H "Authorization: Bearer $ADJUSTMENT_API_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"run_id":"YOUR_COMPLETED_RUN_ID","feedback":"Lift five millimeters higher between the eyes","request_id":"adjustment-001"}'
```

## Actual OpenAI call

`openai_adjustment.py` calls:

```python
client.responses.create(
    model=os.environ['OPENAI_MODEL'],
    store=False,
    instructions=INSTRUCTIONS,
    input=json.dumps(context),
    text={'format': {
        'type': 'json_schema', 'name': 'drawing_adjustment',
        'strict': True, 'schema': SCHEMA,
    }},
)
```

Only routine name, current parameter overrides, and correction text are sent. No robot credentials, raw recordings, or images are sent. This version interprets spoken/text descriptions; image upload analysis is not implemented.

The local planner rechecks type, bounds, geometry and joint speed. The endpoint cannot execute a drawing. A separate `draw` call selects the returned revision. Unsupported requests return clarification rather than generated robot code.

Offline tests use a fake OpenAI client and local HTTP server. Real API credentials, public HTTPS deployment and live voice integration still require setup and testing.

Official reference: https://developers.openai.com/api/docs/guides/structured-outputs
