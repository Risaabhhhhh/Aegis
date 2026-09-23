from base import Request, Context, Response, RequestProfile

def build_request_profile(request: Request, context: Context, response: Response) -> RequestProfile:
    # v1 best-effort implementation
    return RequestProfile(context_available=len(context) > 0)
