# Does removed bio knowledge stay removed? (Plain-language summary)

I ran a small experiment to test something a lot of AI safety researchers worry about:
when companies "unlearn" a dangerous capability from an AI model, does it actually go
away, or does it just get hidden and come back the first time someone fine-tunes the
model for something else entirely?

Here's what I did. I took a small, publicly downloadable language model and used a
published technique to make it forget how to answer a set of test questions about
hazardous biology — not real instructions for anything dangerous, just a standard
benchmark researchers use to measure this kind of knowledge without spreading anything
harmful. Before the treatment, the model got about 68% of these questions right. After
it, that dropped to about 61%. So far, so good — it looked like the "forgetting" worked.

Then I did the part that mattered: I fine-tuned the model on completely ordinary,
harmless text — the kind of generic instruction-following data anyone might use to
customize a chatbot for their own purposes, nothing to do with biology and nothing
designed to undo the forgetting. Within just 25 rounds of this ordinary training, the
score had already climbed back to 66%. By 200 rounds, it was at 67.5% — almost exactly
back where it started before anything was removed.

In other words: the "forgetting" didn't hold up against completely normal, non-malicious
use of the model. Nobody had to try to break it.

This matters because a lot of the safety promises made about open, downloadable AI models
rest on the idea that you can strip out a dangerous capability and it'll stay stripped.
This small test suggests that, at least for the method I used, that's not really true —
the capability was suppressed, not removed, and a small amount of unrelated fine-tuning
brought it right back. That's exactly the concern that AI safety researcher Stephen Casper
has raised about these techniques, and this project is one small, independently-run
example that lines up with his argument.

A few honest caveats: I used a much smaller model and a stand-in training dataset instead
of the exact one the original method's authors used (theirs required special access I
didn't request), so my numbers are illustrative rather than an exact match to published
results. I also stopped the fine-tuning test at 200 rounds, and the recovery was still
climbing at that point — so in reality, even more of the forgotten knowledge would likely
come back with a bit more training.

One safety note: at no point did this project create or store anything actually
dangerous. The test questions are a measurement tool built by safety researchers, never
real instructions. The training data used to test whether the "forgetting" held up was
ordinary, harmless text. And no version of the model — forgetful or "recovered" — was ever
saved or shared anywhere; every copy existed only temporarily while the experiment ran.
