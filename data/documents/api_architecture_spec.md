# Enterprise REST API Architecture & Developer Specification

## 1. Authentication & Session Management
All programmatic requests to the Enterprise API platform must be authenticated using OAuth 2.0 Bearer JSON Web Tokens (JWT) transmitted in the standard HTTP request header:
`Authorization: Bearer <access_token>`

### Token Expiration & Refresh Rotation Standards
- Access tokens carry a maximum validity lifetime of 60 minutes (3,600 seconds) from time of issuance.
- Refresh tokens carry a 30-day validity lifetime and employ strict single-use refresh token rotation (RTR). Upon exchange, the previous refresh token is immediately invalidated and blacklisted.
- Any attempt to reuse an expired or already-exchanged refresh token triggers automated account suspension and alerts the Security Operations Center.

---

## 2. Rate Limiting & Abuse Prevention
API rate limits are enforced at the edge API gateway tier per client identifier (API Key or Organization ID).

### Tiered Quotas & Throttling Parameters
- **Developer / Standard Tier**: 100 requests per minute with a burst buffer of 20 requests.
- **Enterprise Tier**: 1,000 requests per minute with dedicated gateway ingress and burst buffer of 250 requests.
- When a client breaches the rate limit quota, the gateway immediately returns `HTTP 429 Too Many Requests`.
- Standard rate limit response headers include:
  - `X-RateLimit-Limit`: Maximum requests permitted within the 60-second window.
  - `X-RateLimit-Remaining`: Number of requests remaining in current window.
  - `X-RateLimit-Reset`: Unix epoch timestamp indicating when the quota resets.
  - `Retry-After`: Minimum number of seconds the client must pause before retrying.

---

## 3. Webhook Delivery & Retry Backoff Architecture
Outbound event notifications (e.g., `invoice.processed`, `triage.escalated`, `user.provisioned`) are delivered via HTTP POST webhooks signed with an HMAC-SHA256 signature in the `X-Webhook-Signature` header.

### Retry Schedule & Dead Letter Queue (DLQ)
Receiving endpoints must respond with an HTTP status code in the 2xx range (`200 OK` or `202 Accepted`) within a 5-second timeout window. If the client endpoint times out or returns a non-2xx status code, the automated delivery engine executes exponential backoff across 5 retry attempts:
1. **Attempt 1**: 10 seconds following initial failure
2. **Attempt 2**: 30 seconds following Attempt 1
3. **Attempt 3**: 2 minutes following Attempt 2
4. **Attempt 4**: 10 minutes following Attempt 3
5. **Attempt 5**: 60 minutes following Attempt 4

If all 5 retry attempts are exhausted without success, the webhook event payload is permanently transferred to the Dead Letter Queue (DLQ). The event remains inspectable in the developer portal for 14 calendar days, allowing manual replay.

---

## 4. Error Envelope Specification
All non-2xx responses from the API conform strictly to the standard JSON error envelope:
```json
{
  "error": {
    "code": "RESOURCE_NOT_FOUND",
    "message": "The requested enquiry entity with ID 'enq_999' does not exist.",
    "status": 404,
    "timestamp": "2026-09-17T14:30:00Z",
    "trace_id": "trc_8492048f9a2",
    "documentation_url": "https://developer.enterprise.com/errors/RESOURCE_NOT_FOUND"
  }
}
```
Client SDKs must parse the `error.code` string rather than relying solely on HTTP status codes for exception handling.
