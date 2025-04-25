// A simple data class (optional) for type safety / documentation
class PullRequest {
  constructor({ title, body, labels, state, number, created_at }) {
    this.title = title;
    this.body = body;
    this.labels = labels;
    this.state = state;
    this.number = number;
    this.created_at = created_at;
  }
}

module.exports = PullRequest;
