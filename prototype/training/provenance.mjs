/** Evaluation maps must be unseen by both optimization and model selection. */
export function assertHeldOut(seeds,training={}){
  const seen=new Set();
  for(const key of ['trainSeeds','priorTrainingSeeds','validationSeeds','selectionSeeds']){
    const values=training[key]??[];
    if(!Array.isArray(values)||values.some(seed=>!Number.isInteger(seed)||seed<0||seed>4294967295))
      throw new TypeError(`Invalid ${key} provenance in model.`);
    values.forEach(seed=>seen.add(seed));
  }
  const ranges=[];
  for(const key of ['trainingSeedRanges','priorTrainingSeedRanges','validationSeedRanges','selectionSeedRanges']){
    const values=training[key]??[];
    if(!Array.isArray(values)||values.some(range=>!Array.isArray(range)||range.length!==2||
      !Number.isInteger(range[0])||!Number.isInteger(range[1])||range[0]<0||range[1]>4294967296||range[1]<=range[0]))
      throw new TypeError(`Invalid ${key} provenance in model.`);
    ranges.push(...values);
  }
  const overlap=seeds.filter(seed=>seen.has(seed)||ranges.some(([start,end])=>seed>=start&&seed<end));
  if(overlap.length)throw new Error(`Evaluation seeds were used for training or model selection: ${overlap.join(', ')}. Choose unseen seeds.`);
}
