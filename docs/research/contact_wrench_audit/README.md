# Contact-Wrench Audit Mode

The [standalone calculation note](contact_wrench_audit.tex) describes an opt-in
failure policy for the existing contact-wrench controller. The default retains
its baseline fallback. The strict threshold is an externally selected
numerical tolerance on normalized QP slack, not an anatomical or measured
force limit. A successful solve does not validate the contact model, force
measurements, continuous-time behavior, or the swing.
