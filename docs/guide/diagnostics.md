# Diagnostics

The `baglab.diagnostics` module helps inspect topic timing and message delivery.

## Topic rate

```python
import baglab

rate = baglab.topic_rate(df)
# Returns statistics about the publishing rate
```

## Message gaps

```python
gaps = baglab.message_gaps(df, threshold=0.1)
# Detects periods where messages are missing
```

## Topic delay

```python
delay = baglab.topic_delay(df)
# Measures delay between header.stamp and receive time
```

## Latency chain

```python
chain = baglab.latency_chain(bag, ["/sensor", "/perception", "/planning", "/control"])
# Analyzes end-to-end latency through a processing pipeline
```
