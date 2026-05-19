from tenacity import retry, stop_after_attempt, wait_exponential


def retry_external_call(fn):
    return retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )(fn)


@retry_external_call
def invoke_chain(chain, payload):
    return chain.invoke(payload)
